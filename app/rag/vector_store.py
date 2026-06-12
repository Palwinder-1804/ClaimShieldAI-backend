import uuid
import logging
import asyncio
from typing import List, Optional, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from app.core.config import settings
from app.rag.embedder import embeddings_provider

logger = logging.getLogger(__name__)

class QdrantPolicyIndex:
    def __init__(self, manager: 'VectorStoreManager', policy_id: str):
        self.manager = manager
        self.policy_id = policy_id

    def get_all_documents(self) -> List[Document]:
        """
        Retrieves all documents associated with this policy_id from Qdrant using the Scroll API.
        Returns a list of langchain_core.documents.Document objects.
        """
        documents = []
        offset = None
        
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="policy_id",
                    match=models.MatchValue(value=self.policy_id)
                )
            ]
        )
        
        try:
            while True:
                res, next_offset = self.manager.client.scroll(
                    collection_name=self.manager.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in res:
                    payload = point.payload or {}
                    text = payload.get("page_content", "")
                    metadata = payload.get("metadata", {})
                    # Ensure metadata is dict
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata["policy_id"] = self.policy_id
                    documents.append(Document(page_content=text, metadata=metadata))
                    
                if next_offset is None:
                    break
                offset = next_offset
                
            logger.info(f"Retrieved {len(documents)} documents via Qdrant Scroll API for policy_id: {self.policy_id}")
        except Exception as e:
            logger.error(f"Failed to scroll documents for policy_id {self.policy_id} from Qdrant: {e}")
            
        return documents

    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Performs vector similarity search on Qdrant, filtered by this policy_id.
        Returns a list of (Document, score) tuples.
        Note: The score returned here is converted back to a distance format to match
        the expectation of similarity_search_with_score (smaller distance is better).
        We do this via the formula: distance = (1.0 / similarity) - 1.0.
        """
        try:
            query_vector = self.manager.embeddings.embed_query(query)
            
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="policy_id",
                        match=models.MatchValue(value=self.policy_id)
                    )
                ]
            )
            
            results = self.manager.client.query_points(
                collection_name=self.manager.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True
            )
            
            search_results = []
            for point in results.points:
                payload = point.payload or {}
                text = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["policy_id"] = self.policy_id
                doc = Document(page_content=text, metadata=metadata)
                
                cosine_sim = float(point.score)
                # Map cosine similarity (higher is better) to L2-like distance (lower is better)
                # such that vector_sim = 1 / (1 + distance) is exactly equal to cosine_sim.
                if cosine_sim > 0.0001:
                    distance = (1.0 / cosine_sim) - 1.0
                else:
                    distance = 999.0
                
                search_results.append((doc, distance))
                
            return search_results
        except Exception as e:
            logger.error(f"Failed to search similarity in Qdrant for policy_id {self.policy_id}: {e}")
            return []


class VectorStoreManager:
    def __init__(self):
        self.embeddings = embeddings_provider.get_embeddings()
        self.collection_name = settings.QDRANT_COLLECTION
        self._client = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            # Connect using url + api_key or fall back to location (like :memory:)
            if settings.QDRANT_LOCATION:
                logger.info(f"Initializing local Qdrant client at: {settings.QDRANT_LOCATION}")
                self._client = QdrantClient(location=settings.QDRANT_LOCATION)
            elif settings.QDRANT_URL and "your-qdrant-cloud-instance-url" not in settings.QDRANT_URL:
                logger.info(f"Initializing Qdrant Cloud client at URL: {settings.QDRANT_URL}")
                self._client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            else:
                logger.warning("Qdrant URL not configured or has default placeholder. Falling back to local in-memory Qdrant client.")
                self._client = QdrantClient(location=":memory:")
            
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        """
        Creates the collection if it does not already exist in Qdrant.
        """
        try:
            # Check if collection exists
            exists = self._client.collection_exists(collection_name=self.collection_name)
            if not exists:
                logger.info(f"Collection {self.collection_name} does not exist. Creating it.")
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # text-embedding-3-small dimension
                        distance=models.Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection {self.collection_name}: {e}")
            raise

    async def create_and_save_index(self, chunks: List[str], policy_id: str, metadatas: Optional[List[dict]] = None) -> None:
        """
        Embeds document chunks and upserts them to the Qdrant database under the given policy_id.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return

        logger.info(f"Indexing {len(chunks)} chunks into Qdrant collection '{self.collection_name}' for policy/session {policy_id}...")
        
        try:
            # Generate embeddings synchronously (LangChain embeddings call)
            embeddings = await asyncio.to_thread(self.embeddings.embed_documents, chunks)
            
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                # Generate deterministic UUID for each point based on policy_id and index
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{policy_id}_{i}"))
                
                payload = {
                    "policy_id": policy_id,
                    "page_content": chunk,
                }
                if metadatas and i < len(metadatas):
                    payload["metadata"] = metadatas[i]
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} points into Qdrant collection '{self.collection_name}' for policy_id {policy_id}")
        except Exception as e:
            logger.error(f"Failed to index/upsert points to Qdrant: {e}")
            raise

    async def load_index(self, policy_id: str) -> Optional[QdrantPolicyIndex]:
        """
        Loads the Qdrant index scoped to a specific policy_id.
        """
        # Verification check: scroll with limit=1 to verify if any chunks exist for this policy_id.
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="policy_id",
                    match=models.MatchValue(value=policy_id)
                )
            ]
        )
        try:
            res, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            if not res:
                logger.warning(f"No indexed points found in Qdrant for policy_id: {policy_id}")
                return None
                
            logger.info(f"Qdrant index for policy_id {policy_id} loaded successfully.")
            return QdrantPolicyIndex(self, policy_id)
        except Exception as e:
            logger.error(f"Failed to load/verify index for policy_id {policy_id}: {e}")
            return None

    async def delete_index(self, policy_id: str) -> None:
        """
        Deletes all vector points associated with the given policy_id from Qdrant.
        """
        logger.info(f"Deleting points from Qdrant collection '{self.collection_name}' for policy_id {policy_id}...")
        try:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="policy_id",
                        match=models.MatchValue(value=policy_id)
                    )
                ]
            )
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=qdrant_filter
            )
            logger.info(f"Successfully deleted all points for policy_id {policy_id}")
        except Exception as e:
            logger.error(f"Failed to delete points from Qdrant for policy_id {policy_id}: {e}")
            raise

vector_store_manager = VectorStoreManager()

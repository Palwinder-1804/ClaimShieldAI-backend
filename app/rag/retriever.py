import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.rag.vector_store import vector_store_manager

logger = logging.getLogger(__name__)

class PolicyRetriever:
    async def retrieve_clauses(self, query: str, policy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval (0.6 * Vector Similarity + 0.4 * BM25 Keyword Matching).
        Returns top 5 matching policy clauses.
        """
        if not query:
            return []

        # 1. Resolve target policy_id
        target_policy_id = policy_id
        if not target_policy_id:
            try:
                res, _ = await asyncio.to_thread(
                    vector_store_manager.client.scroll,
                    collection_name=vector_store_manager.collection_name,
                    limit=20,
                    with_payload=True,
                    with_vectors=False
                )
                policy_ids = list(set(point.payload.get("policy_id") for point in res if point.payload and "policy_id" in point.payload))
                if policy_ids:
                    target_policy_id = policy_ids[0]
                    logger.info(f"No policy_id provided. Falling back to first found policy_id in Qdrant: {target_policy_id}")
            except Exception as e:
                logger.error(f"Error scrolling Qdrant for fallback policy_ids: {e}")
            
        if not target_policy_id:
            logger.warning("No policy_id provided and no policy data found in Qdrant. Hybrid RAG retrieval is skipped.")
            return []

        # 2. Load Qdrant Index
        db = await vector_store_manager.load_index(target_policy_id)
        if not db:
            logger.warning(f"Could not load Qdrant index for policy_id: {target_policy_id}")
            return []

        # 3. Retrieve all documents from Qdrant store to construct BM25 corpus
        try:
            all_docs = await asyncio.to_thread(db.get_all_documents)
        except Exception as e:
            logger.error(f"Error accessing Qdrant document scroll API: {e}")
            return []

        if not all_docs:
            logger.warning("Loaded Qdrant index is empty.")
            return []

        # Construct BM25 Index
        def init_bm25():
            corpus = [doc.page_content.lower().split() for doc in all_docs]
            return BM25Okapi(corpus)
        bm25 = await asyncio.to_thread(init_bm25)

        # 4. Perform Vector Search (L2 distance scores)
        # Fetching k=20 matches to cross-examine with BM25
        k_count = min(20, len(all_docs))
        vector_results = await asyncio.to_thread(db.similarity_search_with_score, query, k=k_count)
        
        # Convert list of (Doc, L2_distance) to dictionary mappings
        vector_distances = {doc.page_content: dist for doc, dist in vector_results}
        
        # 5. Compute BM25 Scores
        query_tokens = query.lower().split()
        bm25_scores = await asyncio.to_thread(bm25.get_scores, query_tokens)
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0

        # 6. Synthesize Hybrid Score
        hybrid_results = []
        for idx, doc in enumerate(all_docs):
            doc_content = doc.page_content
            
            # Normalise BM25 Score (0 to 1)
            bm25_val = float(bm25_scores[idx]) / max_bm25
            
            # Normalise Qdrant Distance to similarity score (0 to 1)
            # L2 distance is smaller for closer documents. 0 means exact match.
            # Convert to similarity: 1 / (1 + distance)
            if doc_content in vector_distances:
                distance = vector_distances[doc_content]
                vector_sim = 1.0 / (1.0 + float(distance))
            else:
                vector_sim = 0.0  # Outside top vector matches
                
            # Weighted combine
            combined_score = 0.6 * vector_sim + 0.4 * bm25_val
            
            hybrid_results.append({
                "text": doc_content,
                "doc_name": doc.metadata.get("name") or doc.metadata.get("source") or "Policy Document",
                "score": combined_score
            })
            
        # 7. Sort by score descending and take top 5
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        top_matches = hybrid_results[:5]
        
        logger.info(f"Hybrid retrieval finished. Best score: {top_matches[0]['score']:.4f}" if top_matches else "No matches found.")
        return top_matches

policy_retriever = PolicyRetriever()

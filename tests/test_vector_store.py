import pytest
from app.core.config import settings
from app.rag.vector_store import vector_store_manager, QdrantPolicyIndex

@pytest.fixture(autouse=True)
def setup_test_qdrant():
    """
    Fixture to redirect Qdrant vector store manager to an in-memory database
    specifically for unit tests.
    """
    original_location = settings.QDRANT_LOCATION
    settings.QDRANT_LOCATION = ":memory:"
    # Reset internal client so it is re-instantiated in memory
    vector_store_manager._client = None
    
    # Ensure collection exists in-memory
    vector_store_manager.client
    
    yield
    
    # Clean up and restore settings
    settings.QDRANT_LOCATION = original_location
    vector_store_manager._client = None


@pytest.mark.asyncio
async def test_qdrant_indexing_and_loading():
    policy_id = "test-policy-123"
    chunks = [
        "This is the first clause of the policy discussing deductibles.",
        "This is the second clause of the policy regarding coverage limits."
    ]
    metadatas = [
        {"source": "test_doc_1.pdf", "page": 1},
        {"source": "test_doc_1.pdf", "page": 2}
    ]

    # Verify loading a non-existent index returns None
    assert await vector_store_manager.load_index(policy_id) is None

    # Save index
    await vector_store_manager.create_and_save_index(
        chunks=chunks,
        policy_id=policy_id,
        metadatas=metadatas
    )

    # Load index and verify it returns a QdrantPolicyIndex
    db = await vector_store_manager.load_index(policy_id)
    assert db is not None
    assert isinstance(db, QdrantPolicyIndex)
    assert db.policy_id == policy_id


@pytest.mark.asyncio
async def test_qdrant_retrieve_all_documents():
    policy_id_1 = "policy-abc"
    policy_id_2 = "policy-xyz"
    
    await vector_store_manager.create_and_save_index(
        chunks=["ABC insurance clause 1", "ABC insurance clause 2"],
        policy_id=policy_id_1
    )
    
    await vector_store_manager.create_and_save_index(
        chunks=["XYZ coverages info"],
        policy_id=policy_id_2
    )

    db_1 = await vector_store_manager.load_index(policy_id_1)
    assert db_1 is not None
    docs_1 = db_1.get_all_documents()
    assert len(docs_1) == 2
    # Check that documents returned have page_content
    contents_1 = [doc.page_content for doc in docs_1]
    assert "ABC insurance clause 1" in contents_1
    assert "ABC insurance clause 2" in contents_1

    db_2 = await vector_store_manager.load_index(policy_id_2)
    assert db_2 is not None
    docs_2 = db_2.get_all_documents()
    assert len(docs_2) == 1
    assert docs_2[0].page_content == "XYZ coverages info"


@pytest.mark.asyncio
async def test_qdrant_similarity_search_with_metadata_filtering():
    policy_1 = "policy-1"
    policy_2 = "policy-2"
    
    # Store standard texts
    await vector_store_manager.create_and_save_index(
        chunks=["Dental insurance covers teeth cleaning up to $500 yearly."],
        policy_id=policy_1
    )
    await vector_store_manager.create_and_save_index(
        chunks=["Dental insurance covers fillings up to $1000 yearly."],
        policy_id=policy_2
    )

    # Search query "dental insurance teeth" on policy_1
    db_1 = await vector_store_manager.load_index(policy_1)
    results_1 = db_1.similarity_search_with_score("dental insurance teeth", k=5)
    
    assert len(results_1) > 0
    doc, dist = results_1[0]
    assert "teeth cleaning" in doc.page_content
    # Since cosine similarity score > 0, distance should be calculated
    assert dist >= 0.0

    # Search same query on policy_2 - should NOT return the teeth cleaning chunk
    db_2 = await vector_store_manager.load_index(policy_2)
    results_2 = db_2.similarity_search_with_score("dental insurance teeth", k=5)
    
    assert len(results_2) > 0
    doc2, dist2 = results_2[0]
    assert "fillings" in doc2.page_content
    assert "teeth cleaning" not in doc2.page_content


@pytest.mark.asyncio
async def test_qdrant_delete_index():
    policy_id = "temp-policy-to-delete"
    
    await vector_store_manager.create_and_save_index(
        chunks=["Temp content 1", "Temp content 2"],
        policy_id=policy_id
    )

    # Verify index loaded successfully
    assert await vector_store_manager.load_index(policy_id) is not None

    # Delete index
    await vector_store_manager.delete_index(policy_id)

    # Verify loading now returns None
    assert await vector_store_manager.load_index(policy_id) is None

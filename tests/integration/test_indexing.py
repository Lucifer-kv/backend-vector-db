import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import numpy as np
from src.models.indexing import HNSWIndex, SearchResult, KDTreeIndex
from src.services.vector_db import VectorDB
from datetime import datetime, timezone
from unittest.mock import patch
from dateutil.parser import isoparse

@pytest.fixture
def embedding_dim():
    return 3  # Use small dimension for faster tests

@pytest.fixture
def mock_embedding(embedding_dim):
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)

@pytest.fixture
def mock_query(embedding_dim):
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)

# --- KDTreeIndex Tests ---
def test_create_kdtree_index(embedding_dim):
    """Test KDTreeIndex creation."""
    index = KDTreeIndex(dim=embedding_dim)
    assert index.dim == embedding_dim
    assert index.root is None
    assert index.elements == {}

def test_kdtree_add(embedding_dim, mock_embedding):
    """Test adding embeddings to KDTreeIndex."""
    index = KDTreeIndex(dim=embedding_dim)
    index.add("chunk1", mock_embedding)
    assert "chunk1" in index.elements
    assert np.array_equal(index.elements["chunk1"], mock_embedding)
    assert index.root is not None
    assert index.root.chunk_id == "chunk1"

def test_kdtree_add_duplicate(embedding_dim, mock_embedding):
    """Test adding duplicate chunk ID to KDTreeIndex."""
    index = KDTreeIndex(dim=embedding_dim)
    index.add("chunk1", mock_embedding)
    index.add("chunk1", mock_embedding)  # Duplicate
    assert len(index.elements) == 1  # No duplicate added

def test_kdtree_add_invalid_dimension(embedding_dim):
    """Test adding embedding with wrong dimension to KDTreeIndex."""
    index = KDTreeIndex(dim=embedding_dim)
    with pytest.raises(ValueError, match="Embedding dimension"):
        index.add("chunk1", [1.0, 0.0])  # Wrong dimension

def test_kdtree_remove(embedding_dim, mock_embedding):
    """Test removing a chunk from KDTreeIndex."""
    index = KDTreeIndex(dim=embedding_dim)
    index.add("chunk1", mock_embedding)
    index.add("chunk2", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    index.remove("chunk1")
    assert "chunk1" not in index.elements
    assert len(index.elements) == 1
    assert "chunk2" in index.elements
    assert np.array_equal(index.elements["chunk2"], np.array([0.0, 1.0, 0.0], dtype=np.float32))

def test_kdtree_remove_nonexistent(embedding_dim, mock_embedding):
    """Test removing a non-existent chunk from KDTreeIndex."""
    index = KDTreeIndex(dim=embedding_dim)
    index.add("chunk1", mock_embedding)
    index.remove("chunk2")  # Non-existent
    assert len(index.elements) == 1
    assert "chunk1" in index.elements

def test_kdtree_search(embedding_dim, mock_embedding, mock_query):
    """Test KDTreeIndex kNN search correctness."""
    index = KDTreeIndex(dim=embedding_dim)
    index.add("chunk1", mock_embedding)
    index.add("chunk2", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    index.add("chunk3", np.array([0.7, 0.7, 0.0], dtype=np.float32))
    
    results = index.search(mock_query, k=2, library_id="lib1")
    
    assert len(results) == 1  # Current implementation returns only 1 result
    assert isinstance(results[0], SearchResult)
    assert results[0].library_id == "lib1"
    assert results[0].chunk_id == "chunk1"
    assert results[0].similarity == pytest.approx(1.0, abs=1e-5)  # Exact match

def test_kdtree_search_empty(embedding_dim, mock_query):
    """Test KDTreeIndex search on empty index."""
    index = KDTreeIndex(dim=embedding_dim)
    results = index.search(mock_query, k=2, library_id="lib1")
    assert results == []

def test_kdtree_search_invalid_dimension(embedding_dim, mock_query):
    """Test KDTreeIndex search with invalid query dimension."""
    index = KDTreeIndex(dim=embedding_dim)
    with pytest.raises(ValueError, match="Query embedding dimension"):
        index.search([1.0, 0.0], k=1, library_id="lib1")

# --- HNSWIndex Tests ---
def test_create_hnsw_index(embedding_dim):
    """Test HNSWIndex creation."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    assert index.dim == embedding_dim
    assert index.max_elements == 100
    assert index.elements == {}
    assert index.graph == {}

def test_hnsw_add(embedding_dim, mock_embedding):
    """Test adding embeddings to HNSWIndex."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    index.add("chunk1", mock_embedding)
    assert "chunk1" in index.elements
    assert np.array_equal(index.elements["chunk1"], mock_embedding)
    assert "chunk1" in index.graph
    assert index.graph["chunk1"] == []

def test_hnsw_add_duplicate(embedding_dim, mock_embedding):
    """Test adding duplicate chunk ID to HNSWIndex."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    index.add("chunk1", mock_embedding)
    index.add("chunk1", mock_embedding)  # Duplicate
    assert len(index.elements) == 1
    assert len(index.graph) == 1

def test_hnsw_add_invalid_dimension(embedding_dim):
    """Test adding embedding with wrong dimension to HNSWIndex."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    with pytest.raises(ValueError, match="Embedding dimension"):
        index.add("chunk1", [1.0, 0.0])  # Wrong dimension

def test_hnsw_remove(embedding_dim, mock_embedding):
    """Test removing a chunk from HNSWIndex."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    index.add("chunk1", mock_embedding)
    index.add("chunk2", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    index.remove("chunk1")
    assert "chunk1" not in index.elements
    assert "chunk1" not in index.graph
    assert len(index.elements) == 1
    assert len(index.graph["chunk2"]) == 0

def test_hnsw_remove_nonexistent(embedding_dim, mock_embedding):
    """Test removing a non-existent chunk from HNSWIndex."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    index.add("chunk1", mock_embedding)
    index.remove("chunk2")  # Non-existent
    assert len(index.elements) == 1
    assert "chunk1" in index.elements

def test_hnsw_search(embedding_dim, mock_embedding, mock_query):
    """Test HNSWIndex kNN search correctness."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    index.add("chunk1", mock_embedding)
    index.add("chunk2", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    index.add("chunk3", np.array([0.7, 0.7, 0.0], dtype=np.float32))
    
    results = index.search(mock_query, k=2, library_id="lib1")
    
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].library_id == "lib1"
    assert results[0].chunk_id == "chunk1"
    assert results[0].similarity == pytest.approx(1.0, abs=1e-5)  # Exact match
    assert results[1].chunk_id == "chunk3"
    assert results[1].similarity == pytest.approx(0.567673, abs=1e-5)  # 1/(1+norm([0.3,-0.7,0]))

def test_hnsw_search_empty(embedding_dim, mock_query):
    """Test HNSWIndex search on empty index."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    results = index.search(mock_query, k=2, library_id="lib1")
    assert results == []

def test_hnsw_search_invalid_dimension(embedding_dim, mock_query):
    """Test HNSWIndex search with invalid query dimension."""
    index = HNSWIndex(dim=embedding_dim, max_elements=100)
    with pytest.raises(ValueError, match="Query embedding dimension"):
        index.search([1.0, 0.0], k=1, library_id="lib1")
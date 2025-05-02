import pytest
from src.indexing import HNSWIndex, SearchResult

def test_brute_force_search():
    index = HNSWIndex()
    index.add("chunk1", [1.0, 0.0])
    index.add("chunk2", [0.0, 1.0])
    results = index.search([1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk1"
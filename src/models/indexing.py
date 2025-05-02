from typing import List, Dict, Tuple
import numpy as np
import heapq

class SearchResult:
    def __init__(self, library_id: str, chunk_id: str, similarity: float):
        self.library_id = library_id
        self.chunk_id = chunk_id
        self.similarity = similarity

class HNSWIndex:
    def __init__(self, max_elements: int = 10000, dim: int = 1024, m: int = 16, ef_construction: int = 200):
        self.max_elements = max_elements
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.elements: Dict[str, np.ndarray] = {}
        self.graph: Dict[str, List[Tuple[str, float]]] = {}

    def add(self, chunk_id: str, embedding: List[float]):
        if chunk_id in self.elements:
            return
        
        embedding = np.array(embedding, dtype=np.float32)
        if embedding.shape != (self.dim,):
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match expected dimension {self.dim}")
        
        self.elements[chunk_id] = embedding
        
        if not self.graph:
            self.graph[chunk_id] = []
            return
        
        distances = []
        for other_id, other_embedding in self.elements.items():
            if other_id == chunk_id:
                continue
            dist = np.linalg.norm(embedding - other_embedding)
            heapq.heappush(distances, (dist, other_id))
        
        neighbors = []
        while distances and len(neighbors) < self.m:
            dist, other_id = heapq.heappop(distances)
            neighbors.append((other_id, dist))
        
        self.graph[chunk_id] = neighbors
        for other_id, dist in neighbors:
            if other_id not in self.graph:
                self.graph[other_id] = []
            self.graph[other_id].append((chunk_id, dist))
            self.graph[other_id].sort(key=lambda x: x[1])
            if len(self.graph[other_id]) > self.m:
                self.graph[other_id].pop()

    def remove(self, chunk_id: str):
        if chunk_id not in self.elements:
            return
        
        del self.elements[chunk_id]
        
        neighbors = self.graph.pop(chunk_id, [])
        for neighbor_id, _ in neighbors:
            if neighbor_id in self.graph:
                self.graph[neighbor_id] = [(cid, dist) for cid, dist in self.graph[neighbor_id] if cid != chunk_id]

    def search(self, query_embedding: List[float], k: int, library_id: str = "") -> List[SearchResult]:
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.shape != (self.dim,):
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match expected dimension {self.dim}")
        
        if not self.elements:
            return []
        
        distances = []
        for chunk_id, embedding in self.elements.items():
            dist = np.linalg.norm(query_embedding - embedding)
            heapq.heappush(distances, (dist, chunk_id))
        
        results = []
        while distances and len(results) < k:
            dist, chunk_id = heapq.heappop(distances)
            similarity = 1 / (1 + dist)
            results.append(SearchResult(library_id=library_id, chunk_id=chunk_id, similarity=similarity))
        
        return results
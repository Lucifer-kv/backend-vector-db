from typing import List, Dict, Tuple, Optional
import numpy as np
import heapq
import random
import math

class SearchResult:
    def __init__(self, library_id: str, chunk_id: str, similarity: float):
        self.library_id = library_id
        self.chunk_id = chunk_id
        self.similarity = similarity

class KDNode:
    def __init__(self, point: np.ndarray, chunk_id: str, axis: int = 0):
        self.point = point
        self.chunk_id = chunk_id
        self.left = None
        self.right = None
        self.axis = axis

    def __lt__(self, other):
        return self.point[self.axis] < other.point[self.axis]

class KDTreeIndex:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.root = None
        self.elements: Dict[str, np.ndarray] = {}

    def add(self, chunk_id: str, embedding: List[float]):
        if chunk_id in self.elements:
            return
        
        embedding = np.array(embedding, dtype=np.float32)
        if embedding.shape != (self.dim,):
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match expected dimension {self.dim}")
        
        self.elements[chunk_id] = embedding
        
        # Insert into KD-tree
        if not self.root:
            self.root = KDNode(embedding, chunk_id, 0)
        else:
            current = self.root
            depth = 0
            while True:
                axis = depth % self.dim
                if embedding[axis] < current.point[axis]:
                    if current.left is None:
                        current.left = KDNode(embedding, chunk_id, axis)
                        break
                    current = current.left
                else:
                    if current.right is None:
                        current.right = KDNode(embedding, chunk_id, axis)
                        break
                    current = current.right
                depth += 1

    def remove(self, chunk_id: str):
        if chunk_id not in self.elements:
            return
        
        # Simplified: Rebuild the tree after removal
        del self.elements[chunk_id]
        embeddings = list(self.elements.values())
        chunk_ids = list(self.elements.keys())
        self.root = None
        for i, embedding in enumerate(embeddings):
            self.add(chunk_ids[i], embedding)

    def search(self, query_embedding: List[float], k: int, library_id: str = "") -> List[SearchResult]:
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.shape != (self.dim,):
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match expected dimension {self.dim}")
        
        if not self.elements:
            return []

        def _nearest(node: Optional[KDNode], target: np.ndarray, k: int, best: List[Tuple[float, str]], depth: int):
            if node is None:
                return

            axis = depth % self.dim
            diff = target[axis] - node.point[axis]
            close_node, other_node = (node.left, node.right) if diff < 0 else (node.right, node.left)

            _nearest(close_node, target, k, best, depth + 1)

            dist = np.linalg.norm(node.point - target)
            if len(best) < k or dist < best[-1][0]:
                best.append((dist, node.chunk_id))
                best.sort()
                if len(best) > k:
                    best.pop()

            if abs(diff) < best[-1][0]:
                _nearest(other_node, target, k, best, depth + 1)

        best = []
        _nearest(self.root, query_embedding, k, best, 0)
        results = []
        for dist, chunk_id in best:
            similarity = 1.0 / (1.0 + dist)
            results.append(SearchResult(library_id=library_id, chunk_id=chunk_id, similarity=similarity))
        return results

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
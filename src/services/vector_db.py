from typing import List, Dict, Optional, Tuple
from src.models.domain import Library, Document, Chunk
from src.models.indexing import HNSWIndex, KDTreeIndex, SearchResult
import threading

class VectorDB:
    def __init__(self):
        self.libraries: Dict[str, Library] = {}
        self.indexes: Dict[str, any] = {}  # Will store either HNSWIndex or KDTreeIndex
        self.lock = threading.RLock()
        self.indexing_type = "kd_tree"  # Options: "hnsw", "kd_tree"

    def _create_index(self):
        if self.indexing_type == "hnsw":
            return HNSWIndex()
        elif self.indexing_type == "kd_tree":
            return KDTreeIndex()
        else:
            raise ValueError(f"Unsupported indexing type: {self.indexing_type}")

    def create_library(self, library: Library):
        with self.lock:
            if library.id in self.libraries:
                raise ValueError(f"Library with ID {library.id} already exists")
            self.libraries[library.id] = library
            self.indexes[library.id] = self._create_index()

    def get_library(self, library_id: str) -> Optional[Library]:
        with self.lock:
            return self.libraries.get(library_id)

    def update_library(self, library_id: str, metadata: Optional[dict]):
        with self.lock:
            if library_id not in self.libraries:
                raise ValueError(f"Library with ID {library_id} not found")
            if metadata is not None:
                self.libraries[library_id].metadata = metadata

    def delete_library(self, library_id: str):
        with self.lock:
            if library_id not in self.libraries:
                raise ValueError(f"Library with ID {library_id} not found")
            del self.libraries[library_id]
            del self.indexes[library_id]

    def add_document(self, library_id: str, document: Document):
        with self.lock:
            if library_id not in self.libraries:
                raise ValueError(f"Library with ID {library_id} not found")
            for lib_id, lib in self.libraries.items():
                for doc in lib.documents:
                    if doc.id == document.id:
                        raise ValueError(f"Document with ID {document.id} already exists in library {lib_id}")
            self.libraries[library_id].documents.append(document)

    def get_document(self, library_id: str, document_id: str) -> Optional[Document]:
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                return None
            for document in library.documents:
                if document.id == document_id:
                    return document
            return None

    def get_document_by_id(self, document_id: str) -> Optional[tuple[str, Document]]:
        with self.lock:
            for library_id, library in self.libraries.items():
                for document in library.documents:
                    if document.id == document_id:
                        return library_id, document
            return None

    def update_document(self, library_id: str, document_id: str, metadata: Optional[dict]):
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                raise ValueError(f"Library with ID {library_id} not found")
            for document in library.documents:
                if document.id == document_id:
                    if metadata is not None:
                        document.metadata = metadata
                    return
            raise ValueError(f"Document with ID {document_id} not found in library {library_id}")

    def update_document_by_id(self, document_id: str, metadata: Optional[dict]):
        with self.lock:
            for library in self.libraries.values():
                for document in library.documents:
                    if document.id == document_id:
                        if metadata is not None:
                            document.metadata = metadata
                        return
            raise ValueError(f"Document with ID {document_id} not found")

    def delete_document(self, library_id: str, document_id: str):
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                raise ValueError(f"Library with ID {library_id} not found")
            for i, document in enumerate(library.documents):
                if document.id == document_id:
                    library.documents.pop(i)
                    for chunk in document.chunks:
                        self.indexes[library_id].remove(chunk.id)
                    return
            raise ValueError(f"Document with ID {document_id} not found in library {library_id}")

    def delete_document_by_id(self, document_id: str):
        with self.lock:
            for library_id, library in self.libraries.items():
                for i, document in enumerate(library.documents):
                    if document.id == document_id:
                        document = library.documents.pop(i)
                        for chunk in document.chunks:
                            self.indexes[library_id].remove(chunk.id)
                        return
            raise ValueError(f"Document with ID {document_id} not found")

    def add_chunk(self, library_id: str, document_id: str, chunk: Chunk):
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                raise ValueError(f"Library with ID {library_id} not found")
            document = None
            for doc in library.documents:
                if doc.id == document_id:
                    document = doc
                    break
            if document is None:
                raise ValueError(f"Document with ID {document_id} not found in library {library_id}")
            for lib_id, lib in self.libraries.items():
                for doc in lib.documents:
                    for c in doc.chunks:
                        if c.id == chunk.id:
                            raise ValueError(f"Chunk with ID {chunk.id} already exists in library {lib_id}, document {doc.id}")
            document.chunks.append(chunk)
            # Ensure embedding is a list of floats
            try:
                embedding = [float(x) if isinstance(x, (str, int)) else x for x in chunk.embedding]
            except (ValueError, TypeError) as e:
                print(f"Error converting embedding for chunk {chunk.id}: {e}")
                raise ValueError(f"Invalid embedding for chunk {chunk.id}")
            self.indexes[library_id].add(chunk.id, embedding)

    def get_chunk(self, library_id: str, document_id: str, chunk_id: str) -> Optional[Chunk]:
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                return None
            for document in library.documents:
                if document.id == document_id:
                    for chunk in document.chunks:
                        if chunk.id == chunk_id:
                            return chunk
                    return None
            return None

    def get_chunk_by_id(self, chunk_id: str) -> Optional[tuple[str, str, Chunk]]:
        with self.lock:
            for library_id, library in self.libraries.items():
                for document in library.documents:
                    for chunk in document.chunks:
                        if chunk.id == chunk_id:
                            return library_id, document.id, chunk
            return None

    def update_chunk(self, library_id: str, document_id: str, chunk_id: str, updated_chunk: Chunk):
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                raise ValueError(f"Library with ID {library_id} not found")
            for document in library.documents:
                if document.id == document_id:
                    for i, chunk in enumerate(document.chunks):
                        if chunk.id == chunk_id:
                            document.chunks[i] = updated_chunk
                            self.indexes[library_id].remove(chunk_id)
                            try:
                                embedding = [float(x) if isinstance(x, (str, int)) else x for x in updated_chunk.embedding]
                            except (ValueError, TypeError) as e:
                                print(f"Error converting embedding for chunk {updated_chunk.id}: {e}")
                                raise ValueError(f"Invalid embedding for chunk {updated_chunk.id}")
                            self.indexes[library_id].add(updated_chunk.id, embedding)
                            return
                    raise ValueError(f"Chunk with ID {chunk_id} not found in document {document_id}")
            raise ValueError(f"Document with ID {document_id} not found in library {library_id}")

    def update_chunk_by_id(self, chunk_id: str, updated_chunk: Chunk):
        with self.lock:
            for library_id, library in self.libraries.items():
                for document in library.documents:
                    for i, chunk in enumerate(document.chunks):
                        if chunk.id == chunk_id:
                            document.chunks[i] = updated_chunk
                            self.indexes[library_id].remove(chunk_id)
                            try:
                                embedding = [float(x) if isinstance(x, (str, int)) else x for x in updated_chunk.embedding]
                            except (ValueError, TypeError) as e:
                                print(f"Error converting embedding for chunk {updated_chunk.id}: {e}")
                                raise ValueError(f"Invalid embedding for chunk {updated_chunk.id}")
                            self.indexes[library_id].add(updated_chunk.id, embedding)
                            return
            raise ValueError(f"Chunk with ID {chunk_id} not found")

    def delete_chunk(self, library_id: str, document_id: str, chunk_id: str):
        with self.lock:
            library = self.libraries.get(library_id)
            if library is None:
                raise ValueError(f"Library with ID {library_id} not found")
            for document in library.documents:
                if document.id == document_id:
                    for i, chunk in enumerate(document.chunks):
                        if chunk.id == chunk_id:
                            document.chunks.pop(i)
                            self.indexes[library_id].remove(chunk_id)
                            return
                    raise ValueError(f"Chunk with ID {chunk_id} not found in document {document_id}")
            raise ValueError(f"Document with ID {document_id} not found in library {library_id}")

    def delete_chunk_by_id(self, chunk_id: str):
        with self.lock:
            for library_id, library in self.libraries.items():
                for document in library.documents:
                    for i, chunk in enumerate(document.chunks):
                        if chunk.id == chunk_id:
                            document.chunks.pop(i)
                            self.indexes[library_id].remove(chunk_id)
                            return
            raise ValueError(f"Chunk with ID {chunk_id} not found")

    def search(self, library_id: str, query_embedding: List[float], k: int) -> List[SearchResult]:
        with self.lock:
            if library_id not in self.libraries:
                raise ValueError(f"Library with ID {library_id} not found")
            return self.indexes[library_id].search(query_embedding, k, library_id=library_id)

    def search_all_libraries(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        with self.lock:
            all_results = []
            for library_id in self.libraries:
                results = self.indexes[library_id].search(query_embedding, k, library_id=library_id)
                all_results.extend(results)
            all_results.sort(key=lambda x: x.similarity, reverse=True)
            return all_results[:k]

    def get_db(self):
        return self

def get_db():
    return VectorDB()
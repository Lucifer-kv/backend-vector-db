from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from ..models.domain import DocumentMetadata, ChunkMetadata,Chunk

class LibraryCreate(BaseModel):
    name: str
    metadata: DocumentMetadata

class LibraryUpdate(BaseModel):
    metadata: Optional[DocumentMetadata] = None

class DocumentCreate(BaseModel):
    metadata: DocumentMetadata

class DocumentUpdate(BaseModel):
    metadata: Optional[DocumentMetadata] = None

class ChunkCreate(BaseModel):
    text: str
    metadata: ChunkMetadata

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[ChunkMetadata] = None

class MetadataFilters(BaseModel):
    name: Optional[str] = None
    createdAfter: Optional[str] = None

class SearchRequest(BaseModel):
    query_text: str
    metadata_filters: Optional[MetadataFilters] = None

class GlobalSearchResult(BaseModel):
    library_id: str
    document_id: str
    chunk: Chunk
    similarity: float
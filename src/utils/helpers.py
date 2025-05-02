from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

class ChunkMetadata(BaseModel):
    name: str
    createdAt: datetime

class DocumentMetadata(BaseModel):
    name: str
    createdAt: datetime

class Chunk(BaseModel):
    id: str = None
    text: str
    embedding: List[float]
    metadata: ChunkMetadata

    def __init__(self, **data):
        if "id" not in data or data["id"] is None:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)

class Document(BaseModel):
    id: str = None
    chunks: List[Chunk]
    metadata: DocumentMetadata

    def __init__(self, **data):
        if "id" not in data or data["id"] is None:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)

class Library(BaseModel):
    id: str
    name: str
    documents: List[Document]
    metadata: DocumentMetadata
    created_at: datetime
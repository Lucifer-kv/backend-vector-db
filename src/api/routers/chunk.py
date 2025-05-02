from fastapi import APIRouter, Depends
from ...models.domain import Chunk
from ...services.vector_db import VectorDB
from ...services.embedding import generate_embedding
from ...api.schemas import ChunkCreate, ChunkUpdate
from ...api.exceptions import NotFoundException, BadRequestException
from ...api.dependencies import get_db
from datetime import datetime

router = APIRouter(prefix="/chunks", tags=["chunks"])

@router.post("/")
async def add_chunk(library_id: str, document_id: str, chunk_data: ChunkCreate, db: VectorDB = Depends(get_db)):
    try:
        metadata = chunk_data.metadata
        if not metadata.createdAt:
            metadata.createdAt = datetime.utcnow()
        
        try:
            embedding = generate_embedding(chunk_data.text)
        except ValueError as e:
            print(f"Failed to generate embedding: {e}")
            embedding = [0.0] * 1024
        except Exception as e:
            print(f"Error calling Cohere API for chunk text '{chunk_data.text}': {e}")
            embedding = [0.0] * 1024
        
        chunk = Chunk(
            text=chunk_data.text,
            embedding=embedding,
            metadata=metadata
        )
        db.add_chunk(library_id, document_id, chunk)
        return {"chunk_id": chunk.id}
    except ValueError as e:
        raise BadRequestException(detail=str(e))

@router.get("/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: str, db: VectorDB = Depends(get_db)):
    result = db.get_chunk_by_id(chunk_id)
    if result is None:
        raise NotFoundException("Chunk", chunk_id)
    return result[2]

@router.patch("/{chunk_id}")
async def update_chunk(chunk_id: str, update: ChunkUpdate, db: VectorDB = Depends(get_db)):
    try:
        result = db.get_chunk_by_id(chunk_id)
        if result is None:
            raise NotFoundException("Chunk", chunk_id)
        library_id, document_id, chunk = result
        
        new_text = update.text if update.text is not None else chunk.text
        new_metadata = update.metadata if update.metadata is not None else chunk.metadata
        
        if new_metadata and not new_metadata.createdAt:
            new_metadata.createdAt = datetime.utcnow()
        
        if new_text != chunk.text:
            try:
                embedding = generate_embedding(new_text)
            except ValueError as e:
                print(f"Failed to generate embedding: {e}")
                embedding = [0.0] * 1024
            except Exception as e:
                print(f"Error calling Cohere API for chunk text '{new_text}': {e}")
                embedding = [0.0] * 1024
        else:
            embedding = chunk.embedding
        
        updated_chunk = Chunk(
            id=chunk.id,
            text=new_text,
            embedding=embedding,
            metadata=new_metadata
        )
        db.update_chunk_by_id(chunk_id, updated_chunk)
        return {"message": "Chunk updated"}
    except ValueError as e:
        raise BadRequestException(detail=str(e))

@router.delete("/{chunk_id}")
async def delete_chunk(chunk_id: str, db: VectorDB = Depends(get_db)):
    try:
        db.delete_chunk_by_id(chunk_id)
        return {"message": "Chunk deleted"}
    except ValueError as e:
        raise BadRequestException(detail=str(e))
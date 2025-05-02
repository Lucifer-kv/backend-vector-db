from fastapi import APIRouter, Depends
from ...models.domain import Library
from ...services.vector_db import VectorDB
from ...api.schemas import LibraryCreate, LibraryUpdate
from ...api.exceptions import NotFoundException, BadRequestException
from ...api.dependencies import get_db
import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/libraries", tags=["libraries"])

@router.post("/", response_model=Library)
async def create_library(library: LibraryCreate, db: VectorDB = Depends(get_db)):
    library_obj = Library(
        id=str(uuid.uuid4()),
        name=library.name,
        documents=[],
        metadata=library.metadata,
        created_at=datetime.now(timezone.utc)  # Updated to use timezone-aware datetime
    )
    try:
        db.create_library(library_obj)
        return library_obj
    except ValueError as e:
        raise BadRequestException(detail=str(e))

@router.get("/{library_id}", response_model=Library)
async def get_library(library_id: str, db: VectorDB = Depends(get_db)):
    library = db.get_library(library_id)
    if library is None:
        raise NotFoundException("Library", library_id)
    return library

@router.patch("/{library_id}")
async def update_library(library_id: str, library: LibraryUpdate, db: VectorDB = Depends(get_db)):
    try:
        db.update_library(library_id, library.metadata)
        return {"message": "Library updated"}
    except ValueError as e:
        raise BadRequestException(detail=str(e))

@router.delete("/{library_id}")
async def delete_library(library_id: str, db: VectorDB = Depends(get_db)):
    try:
        db.delete_library(library_id)
        return {"message": "Library deleted"}
    except ValueError as e:
        raise BadRequestException(detail=str(e))
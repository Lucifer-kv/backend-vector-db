from fastapi import APIRouter, Depends
from ...models.domain import Document
from ...services.vector_db import VectorDB
from ...api.schemas import DocumentCreate, DocumentUpdate
from ...api.exceptions import NotFoundException, BadRequestException
from ...api.dependencies import get_db

router = APIRouter(prefix="/documents", tags=["documents"])
@router.post("/", response_model=Document)
async def create_document(library_id: str, document: DocumentCreate, db: VectorDB = Depends(get_db)):
    try:
        doc = Document(chunks=[], metadata=document.metadata)
        db.add_document(library_id, doc)
        return doc
    except ValueError as e:
        raise BadRequestException(detail=str(e))

@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str, db: VectorDB = Depends(get_db)):
    result = db.get_document_by_id(document_id)
    if result is None:
        raise NotFoundException("Document", document_id)
    return result[1]

@router.patch("/{document_id}")
async def update_document(document_id: str, update: DocumentUpdate, db: VectorDB = Depends(get_db)):
    try:
        db.update_document_by_id(document_id, update.metadata)
        return {"message": "Document updated"}
    except ValueError as e:
        raise BadRequestException(detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str, db: VectorDB = Depends(get_db)):
    try:
        db.delete_document_by_id(document_id)
        return {"message": "Document deleted"}
    except ValueError as e:
        raise BadRequestException(detail=str(e))
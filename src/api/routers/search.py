from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from ...services.vector_db import VectorDB
from ...services.embedding import generate_embedding
from ...api.schemas import SearchRequest, GlobalSearchResult
from ...api.exceptions import NotFoundException, BadRequestException, InternalServerErrorException
from ...api.dependencies import get_db
from datetime import datetime

router = APIRouter(tags=["search"])

@router.post("/libraries/{library_id}/search", response_model=Dict[str, str | float])
async def search(library_id: str, request: SearchRequest, db: VectorDB = Depends(get_db)):
    try:
        query_embedding = generate_embedding(request.query_text)
        k = 5
        results = db.search(library_id, query_embedding, k)
        
        library = db.get_library(library_id)
        if library is None:
            raise NotFoundException("Library", library_id)
        
        filtered_results = []
        for result in results:
            chunk = next(
                (c for doc in library.documents for c in doc.chunks if c.id == result.chunk_id),
                None
            )
            if not chunk:
                continue
            
            if request.metadata_filters:
                if request.metadata_filters.name and request.metadata_filters.name.lower() not in chunk.metadata.name.lower():
                    continue
                if request.metadata_filters.createdAfter:
                    query_date = datetime.strptime(request.metadata_filters.createdAfter, "%Y-%m-%d")
                    chunk_date = datetime.strptime(chunk.metadata.createdAt, "%Y-%m-%d") if isinstance(chunk.metadata.createdAt, str) else chunk.metadata.createdAt
                    if chunk_date <= query_date:
                        continue
            
            filtered_results.append((result, chunk))
        
        if not filtered_results:
            return {"text": "", "similarity": 0.0}
        
        max_result, max_chunk = max(filtered_results, key=lambda x: x[0].similarity)
        return {"text": max_chunk.text, "similarity": max_result.similarity}
    except Exception as e:
        raise InternalServerErrorException(detail=f"Unexpected error: {str(e)}")

@router.post("/search", response_model=List[GlobalSearchResult])
async def search_all_libraries(request: SearchRequest, db: VectorDB = Depends(get_db)):
    try:
        try:
            query_embedding = generate_embedding(request.query_text)
        except ValueError as e:
            print(f"Failed to generate embedding: {e}")
            query_embedding = [0.0] * 1024
        except Exception as e:
            print(f"Error calling Cohere API for query '{request.query_text}': {e}")
            query_embedding = [0.0] * 1024
        
        k = 5  # Hardcoded k
        print(f"Number of libraries: {len(db.libraries)}, search_k: {k}")
        total_chunks = sum(len(doc.chunks) for lib in db.libraries.values() for doc in lib.documents)
        print(f"Total chunks across all libraries: {total_chunks}")
        results = db.search_all_libraries(query_embedding, k)
        
        filtered_results = []
        for result in results:
            library = db.get_library(result.library_id)
            if library is None:
                continue
            
            chunk = None
            document_id = None
            for doc in library.documents:
                for c in doc.chunks:
                    if c.id == result.chunk_id:
                        chunk = c
                        document_id = doc.id
                        break
                if chunk:
                    break
            if not chunk:
                continue
            
            if request.metadata_filters:
                
                if request.metadata_filters.name is None or request.metadata_filters.createdAfter is None:
                    continue
                
                query_name = request.metadata_filters.name.lower()
                chunk_name = chunk.metadata.name.lower()
                if query_name not in chunk_name:
                    continue
            
                try:
                    query_date_str = request.metadata_filters.createdAfter
                    query_date = datetime.strptime(query_date_str, "%Y-%m-%d")
                    
                    
                    if isinstance(chunk.metadata.createdAt, str):
                        chunk_date = datetime.strptime(chunk.metadata.createdAt, "%Y-%m-%dT%H:%M:%S.%f")
                    elif isinstance(chunk.metadata.createdAt, datetime):
                        chunk_date = chunk.metadata.createdAt
                    else:
                        raise ValueError("Invalid createdAt format in chunk metadata")
                    
                    if chunk_date.tzinfo is not None:
                        chunk_date = chunk_date.replace(tzinfo=None)
                    if query_date.tzinfo is not None:
                        query_date = query_date.replace(tzinfo=None)
                    
                    if chunk_date <= query_date:
                        continue
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid date format for createdAfter: must be YYYY-MM-DD")
            
            filtered_results.append((result, chunk, document_id))
        
        filtered_results.sort(key=lambda x: x[0].similarity, reverse=True)
        final_results = filtered_results[:k]
        
        return [
            GlobalSearchResult(
                library_id=result.library_id,
                document_id=document_id,
                chunk=chunk,
                similarity=result.similarity
            )
            for result, chunk, document_id in final_results
        ]
    except Exception as e:
        raise InternalServerErrorException(f"Unexpected error: {str(e)}")
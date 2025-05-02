from fastapi import Depends, HTTPException, status
from ..services.vector_db import VectorDB

_db = VectorDB()

def get_db() -> VectorDB:
    return _db
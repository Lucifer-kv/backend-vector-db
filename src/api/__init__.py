from fastapi import FastAPI
from .routers.library import router as library_router
from .routers.document import router as document_router
from .routers.chunk import router as chunk_router
from .routers.search import router as search_router

app = FastAPI(title="Vector DB API", version="1.0.0")

app.include_router(library_router)
app.include_router(document_router)
app.include_router(chunk_router)
app.include_router(search_router)
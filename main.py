import uvicorn
from src.api import app
from src.core.logging import setup_logging

logger = setup_logging()

if __name__ == "__main__":
    logger.info("Starting Vector DB API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
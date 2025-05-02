# Vector DB API

A FastAPI-based backend service for vector database operations, providing efficient storage and retrieval of vector embeddings.

## Features

- FastAPI-based REST API
- Vector database operations
- Docker containerization support
- Kubernetes deployment configuration
- Comprehensive logging system
- Test suite included

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for cluster deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lucifer-kv/backend-vector-db.git
cd backend-vector-db
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Local Development

Run the application using:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

Build and run the Docker container:
```bash
docker build -t vector-db-api .
docker run -p 8000:8000 vector-db-api
```

### Kubernetes Deployment

Deploy using the provided Kubernetes configurations:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Project Structure

```
backend-vector-db/
├── src/                # Source code
├── tests/             # Test files
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration
├── deployment.yaml    # Kubernetes deployment config
├── service.yaml       # Kubernetes service config
└── setup.sh           # Setup script
```

## API Documentation

Once the application is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`

## Testing

Run the test suite:
```bash
pytest tests/
```

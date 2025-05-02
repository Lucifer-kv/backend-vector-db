import requests
from faker import Faker
from datetime import datetime
from ..core.config import settings

NUM_LIBRARIES = 10
DOCS_PER_LIBRARY = 5
CHUNKS_PER_DOCUMENT = 10

fake = Faker()

def create_library(library_num):
    response = requests.post(
        f"{settings.BASE_URL}/libraries/",
        json={
            "name": f"Library {library_num}",
            "metadata": {
                "name": f"Library {library_num}",
                "createdAt": datetime.utcnow().isoformat()
            }
        }
    )
    response.raise_for_status()
    return response.json()["id"]

def create_document(library_id, doc_num):
    response = requests.post(
        f"{settings.BASE_URL}/documents/?library_id={library_id}",
        json={
            "metadata": {
                "name": f"Document {doc_num}",
                "createdAt": datetime.utcnow().isoformat()
            }
        }
    )
    response.raise_for_status()
    return response.json()["id"]

def create_chunk(library_id, document_id, chunk_num):
    random_sentence = fake.sentence()
    response = requests.post(
        f"{settings.BASE_URL}/chunks/?library_id={library_id}&document_id={document_id}",
        json={
            "text": random_sentence,
            "metadata": {
                "name": f"Chunk {chunk_num}",
                "createdAt": datetime.utcnow().isoformat()
            }
        }
    )
    response.raise_for_status()
    return response.json()["chunk_id"]

def main():
    total_chunks = NUM_LIBRARIES * DOCS_PER_LIBRARY * CHUNKS_PER_DOCUMENT
    print(f"Creating {total_chunks} chunks across {NUM_LIBRARIES} libraries...")

    for lib_num in range(NUM_LIBRARIES):
        print(f"Creating library {lib_num + 1}/{NUM_LIBRARIES}...")
        library_id = create_library(lib_num + 1)

        for doc_num in range(DOCS_PER_LIBRARY):
            document_id = create_document(library_id, doc_num + 1)

            for chunk_num in range(CHUNKS_PER_DOCUMENT):
                create_chunk(library_id, document_id, chunk_num + 1)

    print(f"Successfully created {total_chunks} chunks!")

if __name__ == "__main__":
    main()
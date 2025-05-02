import requests
from datetime import datetime
from faker import Faker


fake = Faker()


BASE_URL = "http://localhost:8000"

def create_library(library_num):
    """Create a library with the given number."""
    response = requests.post(
        f"{BASE_URL}/libraries/",
        json={
            "name": f"library-{library_num}",
            "metadata": {
                "name": f"Library {library_num}",
                "createdAt": datetime.utcnow().isoformat()
            }
        }
    )
    response.raise_for_status()
    return response.json()["id"]

def create_document(library_id, doc_num):
    """Create a document in the specified library."""
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/",
        json={
            "id": f"doc-{doc_num}",
            "metadata": {
                "name": f"Document {doc_num}",
                "createdAt": datetime.utcnow().isoformat()
            }
        }
    )
    response.raise_for_status()
    return response.json()["id"]

def create_chunk(library_id, document_id, chunk_num):
    """Create a chunk in the specified document with a random English sentence."""
    random_sentence = fake.sentence()  
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/{document_id}/chunks",
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
    
    for lib_num in range(1, 11):
        try:
            library_id = create_library(lib_num)
            print(f"Created library {lib_num}: {library_id}")

            
            for doc_num in range(1, 6):
                try:
                    document_id = create_document(library_id, doc_num)
                    print(f"  Created document {doc_num} in library {lib_num}: {document_id}")

                    
                    for chunk_num in range(1, 11):
                        try:
                            chunk_id = create_chunk(library_id, document_id, chunk_num)
                            print(f"    Created chunk {chunk_num} in document {doc_num}: {chunk_id}")
                        except requests.exceptions.RequestException as e:
                            print(f"    Failed to create chunk {chunk_num} in document {doc_num}: {e}")
                except requests.exceptions.RequestException as e:
                    print(f"  Failed to create document {doc_num} in library {lib_num}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to create library {lib_num}: {e}")

if __name__ == "__main__":
    main()
import os
import chromadb
from typing import List, Dict, Any
import uuid
import numpy as np

class VectorStore:
    """ Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_docs", persist_directory: str = "vector_store"):
        """Initialize the vector store
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store"""

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector Store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Failed to initialize vector store: {e}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store....")

        # ChromaDB has a max batch size of 5461, so split into smaller batches
        batch_size = 5000
        total_added = 0

        for batch_start in range(0, len(documents), batch_size):
            batch_end = min(batch_start + batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]

            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []

            for i, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{batch_start + i}"
                ids.append(doc_id)

                metadata = dict(doc.metadata)
                metadata['doc_index'] = batch_start + i
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                documents_text.append(doc.page_content)
                embeddings_list.append(embedding.tolist())

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text
                )
                batch_added = len(batch_docs)
                total_added += batch_added
                print(f"Added batch: {batch_start}-{batch_end} ({batch_added} documents)")

            except Exception as e:
                print(f"Failed to add documents [{batch_start}:{batch_end}] to the vector store: {e}")
                raise

        print(f"Successfully added {total_added} documents to vectorstore")
        print(f"Total documents in collection: {self.collection.count()}")





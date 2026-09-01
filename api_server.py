"""
FastAPI backend for RAG pipeline
Provides REST API endpoints for document retrieval and response generation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from src.rag.embedding import EmbeddingPipeline
from src.rag.vectorstore import VectorStore
from src.rag.retriever import RAGRetriever
from src.rag.llm import GroqLLM


# Initialize FastAPI app
app = FastAPI(
    title="ISRO RAG API",
    description="Retrieval-Augmented Generation for ISRO Documents",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (loaded once at startup)
print("🚀 Initializing RAG models...")
embedding_pipeline = EmbeddingPipeline()
vector_store = VectorStore()
retriever = RAGRetriever(vector_store, embedding_pipeline)
llm = GroqLLM()
print("✓ Models loaded and ready!\n")


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0


class RetrievedDocument(BaseModel):
    rank: int
    content: str
    similarity_score: float
    source: str
    page: int


class QueryResponse(BaseModel):
    query: str
    retrieved_documents: List[RetrievedDocument]
    generated_response: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_size: int


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and vectorstore status"""
    return {
        "status": "healthy",
        "vectorstore_size": vector_store.collection.count()
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        query: User question
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        
    Returns:
        Retrieved documents and generated response
    """
    try:
        print(f"📥 Query received: {request.query}")
        
        # Retrieve documents
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        if not results:
            return {
                "query": request.query,
                "retrieved_documents": [],
                "generated_response": "No relevant documents found in the vectorstore."
            }
        
        # Convert to response format
        retrieved_docs = [
            RetrievedDocument(
                rank=r['rank'],
                content=r['content'][:500],  # First 500 chars for preview
                similarity_score=r['similarity_score'],
                source=r['metadata'].get('source', 'Unknown'),
                page=r['metadata'].get('page', 0)
            )
            for r in results
        ]
        
        # Generate response
        context = "\n".join([r['content'] for r in results])
        generated_response = llm.generate_response(request.query, context)
        
        print(f"✓ Response generated ({len(results)} documents retrieved)\n")
        
        return {
            "query": request.query,
            "retrieved_documents": retrieved_docs,
            "generated_response": generated_response
        }
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vectorstore/stats")
async def vectorstore_stats():
    """Get vectorstore statistics"""
    return {
        "total_documents": vector_store.collection.count(),
        "collection_name": vector_store.collection_name,
        "persist_directory": vector_store.persist_directory
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Starting FastAPI server on http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

import gc
from typing import List, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class EmbeddingPipeline:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Loaded Embedding Model on {self.device}: {self.model}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Any], batch_size: int = 16) -> np.ndarray:
        """Generates embeddings in batches to prevent 4GB VRAM Out-Of-Memory crashes.
        """
        texts = [chunk.page_content for chunk in chunks]
        total_chunks = len(texts)
        print(f"Generating embeddings for {total_chunks} chunks")
        
        # Decimal progress bar formatting configuration
        bar_format = "{l_bar}{percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        embeddings_list = []

        with tqdm(total=total_chunks, bar_format=bar_format, desc="Embedding on GPU") as pbar:
            for i in range(0, total_chunks, batch_size):
                batch = texts[i : i + batch_size]
                
                # Use FP16 autocast to save 50% VRAM if running on Nvidia GPU
                if "cuda" in self.device:
                    with torch.amp.autocast(device_type="cuda"):
                        batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                else:
                    batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                
                embeddings_list.append(batch_embeddings)
                pbar.update(len(batch))
                
                # Active VRAM/RAM garbage collection mechanism
                del batch, batch_embeddings
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
                gc.collect()

        # Combine batches smoothly into one final matrix
        embeddings = np.vstack(embeddings_list)
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for raw text strings (for user queries)"""
        print(f"Generating embeddings for {len(texts)} text(s)")
        
        if "cuda" in self.device:
            with torch.amp.autocast(device_type="cuda"):
                embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
        return embeddings
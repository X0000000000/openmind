import os
import uuid
from typing import List
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:
    def __init__(self, persist_dir: str = None, collection_name: str = None, api_key: str = None, base_url: str = None):
        self.persist_dir = Path(persist_dir or os.getenv("CHROMA_DB_DIR", "./data/chroma_db"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "papers")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception:
            # If collection schema changed, recreate
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def add_document(self, text: str, metadata: dict = None) -> None:
        """Add document chunks to vector store."""
        if metadata is None:
            metadata = {}
        
        chunks = self.text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["text"] = chunk
            
            try:
                embedding = self.embeddings.embed_query(chunk)
                doc_id = f"{uuid.uuid4()}"
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[chunk_metadata],
                    ids=[doc_id]
                )
            except Exception as e:
                raise Exception(f"Failed to add document chunk to vector store: {str(e)}")

    def similarity_search(self, query: str, k: int = 5) -> List[dict]:
        """Search for similar documents."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            docs = []
            if results.get("documents") and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    docs.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results.get("distances") and results["distances"] else None
                    })
            return docs
        except Exception as e:
            raise Exception(f"Failed to search vector store: {str(e)}")

    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            raise Exception(f"Failed to clear vector store collection: {str(e)}")
        
    def get_count(self) -> int:
        """Get number of documents in collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import streamlit as st
from qdrant_client import QdrantClient

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = st.secrets["QDRANT_URL"],
        qdrant_api_key: str = st.secrets["QDRANT_API_KEY"],
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, pdf_path: str):
        # Clear the Qdrant collection first
        qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False,
        )
        if qdrant_client.collection_exists(self.collection_name):
            qdrant_client.delete_collection(collection_name=self.collection_name)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the PDF.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")
        try:
            qdrant = Qdrant.from_documents(
                splits,
                self.embeddings,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant Cloud: {e}")
        return "✅ Vector DB Successfully Created and Stored in Qdrant Cloud!"

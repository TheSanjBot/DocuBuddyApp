# chatbot.py
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

class ChatbotManager:
    def __init__(
        self,
        openrouter_api_key =  st.secrets["OPENROUTER_API_KEY"],
        qdrant_url=st.secrets["QDRANT_URL"],
        qdrant_api_key=st.secrets["QDRANT_API_KEY"],
        collection_name: str = "vector_db",
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
    ):
        # Set up embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Connect to Qdrant Cloud
        self.db = Qdrant(
            client=QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=False
            ),
            embeddings=self.embeddings,
            collection_name=collection_name
        )

        # Set up LLM via OpenRouter with DeepSeek R1
        self.llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            model="deepseek/deepseek-r1:free",  # ✅ OpenRouter model name
            temperature=0.7,
        )

        # Prompt Template
        prompt_template = """Use the following pieces of context to answer the question.
If unsure, say you don't know. Don't make up answers.

Context: {context}
Question: {question}

Helpful Answer:
"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # RetrievalQA
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs={"k": 2}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

    def get_response(self, query: str) -> str:
        return self.qa.run(query)

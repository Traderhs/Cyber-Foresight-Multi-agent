import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")


class CyberRAG:
    def __init__(self):
        self.rag = None
        self.working_dir = "./rag_storage"

    async def initialize(self):
        """Initialize RAG system"""
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=os.getenv("LLM_MODEL", "llama3.1:8b"),
            embedding_func=ollama_embed,
        )
        await self.rag.initialize_storages()
        print("RAG system initialization.")

    async def query(self, question, mode="hybrid"):
        """RAG search"""
        if not self.rag:
            return "RAG not initialized."

        result = await self.rag.aquery(
            question,
            param=QueryParam(mode=mode)
        )
        return result

    async def add_conversation_data(self, agent_type, content):
        """Add Agent conversation content to RAG"""
        if self.rag:
            formatted_content = f"**{agent_type} Analysis**: {content}"
            await self.rag.ainsert(formatted_content)

    async def finalize(self):
        """Clean up RAG system"""
        if self.rag:
            await self.rag.finalize_storages()


# Global RAG instance creation
cyber_rag = CyberRAG()

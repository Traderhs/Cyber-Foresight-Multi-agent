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
        self.cti_list = [
            "DDoS-ALL.txt"
#            "Ransomware-ALL.txt"
#            "Malware-ALL.txt"
        ]

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

    async def load_cyber_data(self):
        """Load exchange rate prediction data"""
        data_dir = "./data"
        cti_list = self.cti_list

        # Process data
        for filename in cti_list:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    cti_name = filename.replace('-ALL.txt', '')
                    formatted_data = f"""
**{cti_name} Forecast Data**
{content}"""

                    # Save each file's data individually to RAG
                    await self.rag.ainsert(formatted_data)
                    print(f"{"="*10}\nRAG loaded {filename} exchange rate data")
                    print(f"Load Data & ainsert to RAG:\n{formatted_data}\n{"="*10}")

    async def get_forecast_data(self):
        """Get forecast data from RAG for agent use"""

        if not self.rag:
            return "RAG not initialized."

        cti_list = self.cti_list
        for filename in cti_list:
            cti_name = filename.replace('-ALL.txt', '')
            result = await self.rag.aquery(
                f"Extract {cti_name} Forecast Data",  # Data Extraction Query
                param=QueryParam(mode="hybrid")
            )
        return result

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

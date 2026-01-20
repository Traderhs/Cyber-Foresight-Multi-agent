import os
import csv
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from state import AgentState
from rag import cyber_rag  # Added RAG integration

from prompts_kor import (
    ATTACKER_SYSTEM_PROMPT,
    DEFENDER_SYSTEM_PROMPT,
    MEDIATOR_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
    REGIONAL_SYSTEM_PROMPT,
    FINANCE_BUSINESS_SYSTEM_PROMPT
)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.2)

async def get_rag_context(query_text):
    """Get relevant context from RAG"""
    try:
        if cyber_rag.rag:
            rag_result = await cyber_rag.query(query_text, mode="hybrid")
            return f"\n\n**Reference Materials (RAG-based)**:\n{rag_result}\n"
        return ""
    except Exception as e:
        print(f"RAG search error: {e}")
        return ""

def load_data_node(state: AgentState) -> AgentState:
    """Data Load Node - Load core threat data from TXT file"""
    cti_list = cyber_rag.cti_list
    data_dir = "../B-MTGNN/model/Bayesian/forecast/data/"
    for filename in cti_list:
        cti_name = filename.replace('-ALL.txt', '')
        filepath = os.path.join(data_dir, filename)
        forecast_summary = f"Tratget Cyber Threat: {cti_name}\n\n[{filename} Data]"
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    forecast_summary += f"""
{content}
**Data Format**:
- Data: {cti_name} trends, impacts and predictions. Data period is July 2011 to December 2024 (162 months).
- Forecast: Future {cti_name} forecast values. Forecast data period is January 2025 to December 2027 (36 months).
- 95% Confidence: 95% confidence data is shown for {cti_name} trend prediction.
- Variance: {cti_name} forecast data variance.
"""
                print(f"Data Loaded from {filepath}\n")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"File not found: {filepath}")
    return {
        "forecast_data": forecast_summary,
        "iteration_count": 0,
        "messages": []
    }

async def attacker_node(state: AgentState) -> AgentState:
    print("--- Attacker Turn ---\n")
    llm = get_llm()
    iteration_count = state["iteration_count"]

    rag_context = ""
    try:
        # Simply await the async function directly
        rag_context = await get_rag_context(
            "What are the latest cyber attack trends and attack techniques?"
        )
    except Exception as e:
        print(f"Error getting RAG context for attacker: {e}")

    prompt = f"""
{ATTACKER_SYSTEM_PROMPT.format(
        forecast_data=state["forecast_data"],
        defense_plan=state.get("defense_plan", "None"),
        iteration_count=iteration_count
    )}
{rag_context}

**Request**: Use the above RAG-based reference materials to develop an attack strategy that reflects latest attack trends.
"""

    response = llm.invoke([SystemMessage(content=prompt)])

    # Save Agent result to RAG
    try:
        # Simply await the async function directly
        await cyber_rag.add_conversation_data("Attacker", response.content)
    except Exception as e:
        print(f"Error saving to RAG: {e}")

    return {
        "attack_plan": response.content,
        "messages": [f"**Attacker**: {response.content}"]
    }

async def defender_node(state: AgentState) -> AgentState:
    print("--- Defender Turn ---\n")
    llm = get_llm()
    
    iteration_count = state["iteration_count"]
    
    # RAG-based defense strategy search
    rag_context = ""
    try:
        rag_context = await get_rag_context(
                "What are cyber threat defense strategies and latest security solutions?"
        )
    except Exception as e:
        print(f"Error getting RAG context for defender: {e}")
    
    prompt = f"""
{DEFENDER_SYSTEM_PROMPT.format(
    attack_plan=state["attack_plan"],
    forecast_data=state["forecast_data"],
    iteration_count=iteration_count
)}
{rag_context}
**Request**: Use the above RAG-based reference materials to develop a defense strategy against the attack.
"""

    response = llm.invoke([SystemMessage(content=prompt)])
    
    # Save Agent result to RAG
    try:
        await cyber_rag.add_conversation_data("Defender", response.content)
    except Exception as e:
        print(f"Error saving to RAG: {e}")
    
    return {
        "defense_plan": response.content,
        "messages": [f"**Defender**: {response.content}"]
    }

async def mediator_node(state: AgentState) -> AgentState:
    print("--- Mediator Turn ---\n")
    llm = get_llm()
    
    iteration_count = state["iteration_count"] + 1
    
    # RAG-based analysis for mediator
    rag_context = ""
    try:
        rag_context = await get_rag_context(
                "What are the key cyber threat mitigation strategies and best practices?"
        )
    except Exception as e:
        print(f"Error getting RAG context for mediator: {e}")
    
    prompt = f"""
{MEDIATOR_SYSTEM_PROMPT.format(
    attack_plan=state["attack_plan"],
    defense_plan=state["defense_plan"],
    forecast_data=state["forecast_data"],
    iteration_count=iteration_count
)}
{rag_context}

**Request**: Analyze the attack and defense strategies and provide balanced insights using RAG-based reference materials.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])

    # Save Agent result to RAG
    try:
        await cyber_rag.add_conversation_data("Mediator", response.content)
    except Exception as e:
        print(f"Error saving to RAG: {e}")
    
    return {
        "mediator_review": response.content,
        "iteration_count": iteration_count,
        "messages": [f"**Mediator (Turn {iteration_count})**: {response.content}"]
    }

async def technical_agent_node(state: AgentState) -> AgentState:
    """RAG Enhanced Technical Implementation Agent"""
    print("--- Technical Implementation Agent Turn ---\n")
    llm = get_llm()
    
    # RAG search for technical solutions
    rag_context = ""
    try:
        rag_context = await get_rag_context(
            "What are cyber security technical solutions and implementation methods?"
        )
    except Exception as e:
        print(f"Error getting RAG context for technical agent: {e}")
    
    prompt = f"""
{TECHNICAL_SYSTEM_PROMPT.format(
    mediator_review=state["mediator_review"],
    forecast_data=state["forecast_data"]
)}
{rag_context}

**Request**: Provide technical analysis and implementation strategy from CTO perspective using RAG-based materials.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # Save Agent result to RAG
    try:
        await cyber_rag.add_conversation_data("Technical Implementation Agent", response.content)
    except Exception as e:
        print(f"Error saving to RAG: {e}")
    
    return {
        "technical_analysis": response.content,
        "messages": [f"**Technical Implementation Agent**: {response.content}"]
    }

async def regional_agent_node(state: AgentState) -> AgentState:
    """RAG Enhanced Regional Agent - Regional regulations and compliance advice"""
    print("--- Regional Agent Turn ---\n")
    llm = get_llm()
    
    # RAG search for regional compliance and regulations
    rag_context = ""
    try:
        rag_context = await get_rag_context(
                "What are regional cybersecurity regulations, compliance requirements, and legal considerations?"
        )
    except Exception as e:
        print(f"Error getting RAG context for regional agent: {e}")
    
    prompt = f"""
{REGIONAL_SYSTEM_PROMPT.format(
    technical_analysis=state["technical_analysis"],
    forecast_data=state["forecast_data"]
)}
{rag_context}

**Request**: Provide regional compliance and regulatory strategy using RAG-based reference materials.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # Save Agent result to RAG
    try:
        await cyber_rag.add_conversation_data("Regional Agent", response.content)
    except Exception as e:
        print(f"Error saving to RAG: {e}")
    
    return {
        "regional_strategy": response.content,
        "messages": [f"**Regional Agent**: {response.content}"]
    }

async def finance_business_agent_node(state: AgentState) -> AgentState:
    """RAG Enhanced Finance-Business Agent - CFO/COO perspective financial and operational planning"""
    print("--- Finance-Business Agent Turn ---\n")
    llm = get_llm()
    history = "\n\n".join(state["messages"])
    
    # RAG search for financial and business impact analysis
    rag_context = ""
    try:
        rag_context = await get_rag_context(
                "What are the financial and business impacts of cybersecurity investments and risk management?"
        )
    except Exception as e:
        print(f"Error getting RAG context for finance-business agent: {e}")
    
    prompt = f"""
{FINANCE_BUSINESS_SYSTEM_PROMPT.format(
    technical_analysis=state["technical_analysis"],
    regional_strategy=state["regional_strategy"],
    messages=history,
    forecast_data=state["forecast_data"]
)}
{rag_context}

**Request**: Provide financial and business strategy from CFO/COO perspective using RAG-based reference materials.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # Save Agent result to RAG
    try:
        await cyber_rag.add_conversation_data("Finance-Business Agent", response.content)
    except Exception as e:
        print(f"Error saving to RAG: {e}")
    
    return {
        "finance_business_plan": response.content,
        "messages": [f"**Finance-Business Agent (Final Report)**: {response.content}"]
    }
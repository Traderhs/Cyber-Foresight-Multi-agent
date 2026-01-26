# Multi-Agent Cybersecurity Analysis System

This repository contains a multi-agent collaborative framework for developing cyber threat prediction and response strategies. Six specialized agents collaborate through Retrieval-Augmented Generation (RAG) technology to establish comprehensive cybersecurity strategies. The system operates through a structured 4-phase process: data loading, collaborative discussion, expert analysis, and final report generation.

## Background and Related Work

Cybersecurity strategy development often relies on siloed expert analysis, which may overlook complex interdependencies. Multi-agent systems enable collaborative decision-making, with applications in cybersecurity for balanced threat analysis. Retrieval-Augmented Generation (RAG) enhances knowledge-intensive tasks, which we adapt for real-time cybersecurity insights.

Our framework uniquely combines role-based agent specialization with RAG-enhanced collaborative analysis.

## System Overview

The Multi-Agent system is a collaborative AI agent network built on LangGraph. Each agent performs specific roles and conducts analysis based on the latest cybersecurity knowledge and prediction data using Retrieval-Augmented Generation (RAG) technology.

### Key Features
- **Collaborative Decision Making**: Balanced strategy development through attacker-defender-mediator discussions
- **RAG Integration**: Real-time knowledge retrieval and augmented generation
- **Role-based Specialization**: Analysis by each agent's specialized field
- **Dynamic Workflow**: Iterative discussion mechanism for consensus building

## Methodology

### System Architecture

#### Core Components

1. **Main Execution File (`main.py`)**: Responsible for system initialization and execution, RAG system initialization, prediction data loading, log management, and asynchronous environment configuration.

2. **Graph Workflow (`graph.py`)**: Defines agent interactions using LangGraph, including node connections, conditional routing, maximum iteration limits, and consensus logic.

3. **Agent Nodes (`nodes.py`)**: Implements six specialized agents:
   - **Attacker**: Develops attack scenarios and analyzes vulnerabilities
   - **Defender**: Formulates defense strategies and applies mitigation technologies
   - **Mediator**: Mediates discussions and builds consensus
   - **Technical Agent**: Plans technical implementations
   - **Regional Agent**: Develops regional compliance strategies
   - **Finance-Business Agent**: Analyzes financial and business impacts

4. **State Management (`state.py`)**: Defines LangGraph state schema, manages data sharing between agents, and tracks message history and iteration counts.

5. **RAG System (`rag.py`)**: Implements knowledge retrieval and augmentation using **LightRAG**, storing cyber threat data and conversation content with hybrid search capabilities.

6. **Prompt Management (`prompts.py`)**: Defines role-specific prompts and behavior guidelines for each agent.

### Operation Process

The system operates through four phases:

#### Phase 1: Data Loading and Initialization
Load prediction data, initialize RAG system, and configure environment settings.

#### Phase 2: Collaborative Discussion
Attacker analyzes prediction data to develop attack scenarios. Defender formulates defense strategies. Mediator evaluates both perspectives and determines consensus.

#### Phase 3: Expert Analysis
Technical Agent provides implementation planning. Regional Agent addresses compliance requirements. Finance-Business Agent evaluates financial and organizational impacts.

#### Phase 4: Final Report Generation
Synthesizes all agent analyses into a comprehensive cybersecurity strategy.

## Data Flow

### Input Data
- **Prediction Data**: Cyber threat and mitigation technology trend predictions
- **RAG Knowledge Base**: Latest cybersecurity knowledge and research materials
- **Environment Configuration**: Ollama model, LLM model, API keys, etc.

### Output Results
- **Log Files**: Records of each agent's analysis process (`./Log/`)
- **Final Report**: Comprehensive cybersecurity strategy (`./Report/`)
- **RAG Storage**: Continuous learning of analysis results (`./rag_storage/`)

## Technology Stack

- **Agent Framework**: LangChain, LangGraph, LightRAG
- **Language Model**: Ollama (Llama 3.1 8B)
- **Programming Language**: Python 3.10+
- **Asynchronous Processing**: asyncio
- **Environment Management**: python-dotenv

## Agent Roles Details

### Attacker
- **Role**: Elite Black Hat Hacker
- **Responsibility**: Vulnerability analysis and attack scenario development
- **Output**: Specific attack methodologies and expected impact

### Defender
- **Role**: Chief Information Security Officer (CISO)
- **Responsibility**: Defense-in-Depth strategy formulation
- **Output**: Mitigation technology-based defense plan

### Mediator
- **Role**: Neutral Senior Security Analyst
- **Responsibility**: Strategy evaluation and consensus building
- **Output**: Residual risk assessment and decision making

### Technical Agent
- **Role**: Chief Technology Officer (CTO)
- **Responsibility**: Technical architecture and implementation planning
- **Output**: Detailed technical roadmap and specifications

### Regional Agent
- **Role**: Regional Security Compliance Expert
- **Responsibility**: Regional regulatory compliance and localization
- **Output**: Regulatory compliance strategy and regional adjustments

### Finance-Business Agent
- **Role**: Chief Financial Officer / Operating Officer
- **Responsibility**: Financial planning and organizational change management
- **Output**: ROI analysis and execution strategy

## Conclusion

This multi-agent framework provides a novel approach to cybersecurity strategy development by integrating role-based specialization with RAG-enhanced collaborative analysis. The system's dynamic workflow enables comprehensive threat mitigation strategies.

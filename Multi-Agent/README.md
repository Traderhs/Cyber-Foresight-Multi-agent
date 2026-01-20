# Multi-Agent Cybersecurity Analysis System

This system is a multi-agent collaborative framework for developing cyber threat prediction and response strategies. Based on B-MTGNN model prediction data, attacker, defender, mediator, and expert agents collaborate to establish comprehensive cybersecurity strategies.

## System Overview

The Multi-Agent system is a collaborative AI agent network built on LangGraph. Each agent performs specific roles and conducts analysis based on the latest cybersecurity knowledge and prediction data using Retrieval-Augmented Generation (RAG) technology.

### Key Features
- **Collaborative Decision Making**: Balanced strategy development through attacker-defender-mediator discussions
- **RAG Integration**: Real-time knowledge retrieval and augmented generation
- **Role-based Specialization**: Analysis by each agent's specialized field
- **Dynamic Workflow**: Iterative discussion mechanism for consensus building

## System Architecture

### Core Components

#### 1. Main Execution File (`main.py`)
- Responsible for system initialization and execution
- RAG system initialization
- Log and result storage management
- Asynchronous execution environment configuration

#### 2. Graph Workflow (`graph.py`)
- Agent interaction definition using LangGraph
- Node connection and conditional routing setup
- Maximum iteration count and consensus logic implementation

#### 3. Agent Nodes (`nodes.py`)
Implementation of 6 agents:
- **Attacker**: Attack scenario development and vulnerability analysis
- **Defender**: Defense strategy formulation and mitigation technology application
- **Mediator**: Discussion mediation and consensus building
- **Technical Agent**: Technical implementation planning
- **Regional Agent**: Regional regulatory compliance strategy
- **Finance-Business Agent**: Financial and business strategy development

#### 4. State Management (`state.py`)
- LangGraph state schema definition
- Data sharing and state tracking between agents
- Message history and iteration count management

#### 5. RAG System (`rag.py`)
- Knowledge retrieval and augmentation based on LightRAG
- Storage of cyber threat data and conversation content
- Support for hybrid search mode

#### 6. Prompt Management (`prompts.py`)
- Specialized prompt definition for each agent
- Role-based behavior guidelines and output format specification

## Operation Process

### Phase 1: Data Loading and Initialization
```
Load Data Node → RAG System Initialization → Prediction Data Loading
```

### Phase 2: Collaborative Discussion
```
Attacker → Defender → Mediator
```
- Attacker: Analyzes prediction data to develop attack scenarios
- Defender: Formulates defense strategies against attacks
- Mediator: Evaluates both strategies and determines consensus

### Phase 3: Expert Analysis
```
Technical Agent → Regional Agent → Finance-Business Agent
```
- Technical Agent: Technical implementation planning from CTO perspective
- Regional Agent: Regional regulatory compliance strategy
- Finance-Business Agent: Financial and business strategy from CFO/COO perspective

### Phase 4: Final Report Generation
- Synthesizes all agent analysis results
- Presents actionable cybersecurity strategy

## Data Flow

### Input Data
- **B-MTGNN Prediction Data**: Cyber threat and mitigation technology trend predictions
- **RAG Knowledge Base**: Latest cybersecurity knowledge and research materials
- **Environment Configuration**: Ollama model, LLM model, API keys, etc.

### Output Results
- **Log Files**: Records of each agent's analysis process (`./Log/`)
- **Final Report**: Comprehensive cybersecurity strategy (`./Report/`)
- **RAG Storage**: Continuous learning of analysis results (`./rag_storage/`)

## Technology Stack

- **Framework**: LangChain, LangGraph, LightRAG
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

## Research Contributions and Applications

### Academic Contributions
- Application of multi-agent systems to cybersecurity strategy development
- Verification of RAG technology's security analysis enhancement effects
- Proactive security strategy development methodology using prediction data

### Practical Applications
- Enterprise cybersecurity strategy formulation
- Security investment priority determination
- Regulatory compliance strategy development
- Crisis response planning

## Future Research Directions

- **Scalability Enhancement**: Addition of more agent roles
- **Real-time Adaptation**: Real-time response to dynamic threat environments
- **Multi-modal Integration**: Integration of various data sources
- **Human-AI Collaboration**: Development of collaboration interfaces with security experts
# Advanced Multi-Agent Research System

An autonomous structured research platform built using Retrieval-Augmented Generation (RAG), vector memory (FAISS), multi-agent orchestration, and iterative critique–refinement workflows.

This system decomposes research into structured planning, web retrieval, contextual embedding, analytical drafting, critique evaluation, refinement, and conversational interaction.


## System Overview

The Advanced Multi-Agent Research System is designed to simulate a structured academic research workflow rather than generating generic AI responses.

It integrates:

- Multi-agent task decomposition
- Web retrieval via SerpAPI
- Vector memory using FAISS
- Retrieval-Augmented Generation (RAG)
- Critique and refinement loops
- Assistant-based contextual interaction
- Streamlit-based professional interface


## Architecture

The system follows a modular pipeline architecture:

 <img width="7811" height="8192" alt="architecture_diagram" src="https://github.com/user-attachments/assets/1bba5392-cc8b-4f3e-adae-f7b875c3da42" />


Each agent performs a dedicated role to maintain separation of concerns and architectural clarity.


## Core Components

### 1️⃣ Planner Agent
- Generates structured research plans
- Produces search queries and focus areas
- Enforces JSON output format

### 2️⃣ Retrieval Layer
- Real-time web search using SerpAPI
- Extracts structured organic results
- Summarizes analytical insights

### 3️⃣ RAG Memory System
- Embeds summaries using SentenceTransformers
- Normalizes vectors
- Stores embeddings in FAISS
- Retrieves top-k relevant context

### 4️⃣ Draft Generator
- Produces structured analytical reports
- Grounded strictly in retrieved context
- Maintains academic tone

### 5️⃣ Critic Agent
- Evaluates logical gaps
- Identifies structural weaknesses
- Suggests improvements

### 6️⃣ Improver Agent
- Refines draft using critique
- Produces submission-ready report
- Removes meta-commentary

### 7️⃣ RAG Assistant (Optional Tab)
- Allows interactive Q&A
- Uses final report as context
- Enables contextual exploration


## Interface

Built using Streamlit with:

- Custom CSS styling
- Professional layout
- Multi-tab navigation
- Expandable debug outputs
- Justified academic formatting


## Tech Stack

- Python
- Streamlit
- OpenRouter API (LLM access)
- SerpAPI (Web search)
- FAISS (Vector similarity search)
- SentenceTransformers
- NumPy
- dotenv


## Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/vaibhavr54/advanced-multi-agent-research-system.git
cd advanced-multi-agent-research-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Environment Variables

Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
SERPAPI_API_KEY=your_key_here
```

### 5️⃣ Run Application

```bash
streamlit run research_agent.py
```


## Evaluation Strategy

The system evaluates performance through:

- Groundedness checks
- Retrieval precision
- Hallucination reduction
- Critique-based improvement comparison
- Structured reasoning transparency


## Future Scope

- Academic database integration (ArXiv, PubMed)
- Hybrid retrieval (dense + sparse)
- Persistent vector storage
- Citation generation
- Automated evaluation metrics
- Multi-agent debate architecture
- Fine-tuned domain-specific planners
- PDF export functionality
- Deployment on cloud platforms


## Why This Project Matters

Most AI tools generate responses directly.

This system instead:

- Plans before writing
- Retrieves before reasoning
- Critiques before finalizing
- Separates roles across agents
- Maintains architectural transparency

It demonstrates applied knowledge of:

- RAG pipelines
- Vector search
- Multi-agent orchestration
- LLM prompting
- AI system design


## Topics

```
rag
multi-agent
llm
faiss
streamlit
retrieval-augmented-generation
ai-research
semantic-search
openrouter
```


## 📜 License

This project is open-source and available under the MIT License.


#### Made with ❤️ by Vaibhav Rakshe

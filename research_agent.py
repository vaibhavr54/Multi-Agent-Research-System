import os
import json
import time
import requests
import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Advanced Multi-Agent Research System",
    layout="wide"
)

# ==========================================================
# PROFESSIONAL STYLING
# ==========================================================

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background-color: #0f172a;
    color: white;
}

.block-container {
    max-width: 1100px;
    margin: auto;
}

.navbar {
    padding: 30px 0;
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    border-bottom: 1px solid #1e293b;
}

.hero {
    text-align: center;
    padding: 30px 0 20px 0;
}

.hero h1 {
    font-size: 26px;
    font-weight: 500;
    color: #cbd5e1;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 40px;
    margin-bottom: 20px;
}

.report-content {
    background-color: white;
    color: black;
    padding: 40px;
    border-radius: 6px;
    text-align: justify;
    line-height: 1.8;
    font-size: 16px;
    margin-top: 30px;
}

.report-content h1,
.report-content h2,
.report-content h3 {
    text-align: left;
    margin-top: 25px;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    border: none;
}

.stTextInput>div>div>input {
    background-color: #1e293b;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="navbar">Advanced Multi-Agent Research System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Architected Intelligence for Structured Research</h1>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# NAVIGATION
# ==========================================================

page = st.radio(
    "Navigation",
    [
        "Research System",
        "Interactive Assistant",
        "Architecture",
        "Why Not ChatGPT?",
        "Evaluation Strategy",
        "Future Scope"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

# ==========================================================
# ENV SETUP
# ==========================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENROUTER_API_KEY or not SERPAPI_API_KEY:
    st.error("API keys missing.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "meta-llama/llama-3-8b-instruct"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ==========================================================
# UTIL FUNCTIONS
# ==========================================================

def call_llm(system_prompt, user_prompt, temperature=0.6):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def call_llm_json(system_prompt, user_prompt, temperature=0.3):
    for _ in range(3):
        response = call_llm(system_prompt, user_prompt, temperature)
        try:
            return json.loads(response)
        except:
            continue
    return None


def serpapi_search(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 5
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    data = response.json()

    results = []
    if "organic_results" in data:
        for result in data["organic_results"][:5]:
            results.append({
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "link": result.get("link")
            })
    return results

# ==========================================================
# RAG MEMORY
# ==========================================================

class ResearchMemory:
    def __init__(self):
        self.documents = []
        self.vectors = []
        self.index = None

    def add_documents(self, texts):
        for text in texts:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            for chunk in chunks:
                self.documents.append(chunk)
                vec = embedding_model.encode(chunk)
                vec = vec / np.linalg.norm(vec)
                self.vectors.append(vec)

        if self.vectors:
            dimension = len(self.vectors[0])
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(np.array(self.vectors))

    def retrieve(self, query, k=5):
        if not self.index:
            return []
        query_vec = embedding_model.encode(query)
        query_vec = query_vec / np.linalg.norm(query_vec)
        D, I = self.index.search(np.array([query_vec]), k)
        return [self.documents[i] for i in I[0]]

# ==========================================================
# AGENTS
# ==========================================================

class PlannerAgent:
    def run(self, topic):
        system_prompt = """
        Respond ONLY in valid JSON format:

        {
            "topic": "...",
            "search_queries": ["..."],
            "focus_areas": ["..."]
        }
        """

        user_prompt = f"""
        Create a structured research plan for the topic:
        {topic}

        Include:
        - 3-5 search queries
        - 3-5 focus areas
        """

        result = call_llm_json(system_prompt, user_prompt)

        if result:
            return result
        else:
            return {
                "topic": topic,
                "search_queries": [topic],
                "focus_areas": ["General overview"]
            }


class CriticAgent:
    def run(self, report):
        prompt = f"""
        Critically evaluate this research report.
        Identify weaknesses, logical gaps, and areas to strengthen.

        Report:
        {report}
        """
        return call_llm("You are an academic reviewer.", prompt, 0.3)


class ImproverAgent:
    def run(self, report, critique):
        prompt = f"""
        Improve the following report using the critique provided.

        Report:
        {report}

        Critique:
        {critique}
        """
        return call_llm("You are a senior academic editor.", prompt, 0.6)

# ==========================================================
# RESEARCH SYSTEM PAGE
# ==========================================================

if page == "Research System":

    st.markdown('<div class="section-title">Execute Structured Research Workflow</div>', unsafe_allow_html=True)

    topic = st.text_input("Enter Research Topic")
    show_debug = st.checkbox("Show Intermediate Outputs")

    if st.button("Start Research") and topic:

        planner = PlannerAgent()
        critic = CriticAgent()
        improver = ImproverAgent()
        memory = ResearchMemory()

        status = st.empty()

        status.markdown("Generating structured research plan...")
        plan = planner.run(topic)
        status.empty()

        st.markdown("### Structured Research Plan")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Search Queries")
            for q in plan["search_queries"]:
                st.markdown(f"- {q}")

        with col2:
            st.markdown("Focus Areas")
            for f in plan["focus_areas"]:
                st.markdown(f"- {f}")

        status.markdown("Conducting web retrieval and summarization...")

        summaries = []
        for q in plan["search_queries"]:
            results = serpapi_search(q)
            if results:
                summary = call_llm(
                    "Summarize and extract key analytical insights.",
                    json.dumps(results, indent=2),
                    0.5
                )
                summaries.append(summary)

        status.markdown("Embedding and retrieving relevant context...")

        if summaries:
            memory.add_documents(summaries)
            retrieved = memory.retrieve(topic)
        else:
            retrieved = []

        status.markdown("Generating draft report...")

        if retrieved:
            draft = call_llm(
                "Write a structured analytical research report grounded strictly in context.",
                "\n\n".join(retrieved),
                0.7
            )
        else:
            draft = "Insufficient retrieval data to generate report."

        status.markdown("Performing critique...")
        critique = critic.run(draft)

        status.markdown("Refining report...")
        final = improver.run(draft, critique)

        status.empty()

        st.session_state["final_memory"] = memory
        st.session_state["final_report"] = final

        st.markdown(
            f'<div class="report-content">{final}</div>',
            unsafe_allow_html=True
        )

        if show_debug:
            with st.expander("Draft Report"):
                st.markdown(draft)
            with st.expander("Critique Output"):
                st.markdown(critique)

# ==========================================================
# INTERACTIVE ASSISTANT (RAG CHATBOT)
# ==========================================================

elif page == "Interactive Assistant":

    st.markdown('<div class="section-title">Interactive Research Assistant</div>', unsafe_allow_html=True)

    if "final_memory" not in st.session_state or "final_report" not in st.session_state:
        st.info("Please generate a research report first.")
    else:

        memory = st.session_state["final_memory"]
        report_text = st.session_state["final_report"]

        left, right = st.columns([1.2, 1])

        # -------------------------
        # LEFT SIDE – REPORT VIEW
        # -------------------------
        with left:
            st.markdown("### Generated Research Report")
            st.markdown(
                f'<div class="report-content">{report_text}</div>',
                unsafe_allow_html=True
            )

        # -------------------------
        # RIGHT SIDE – CHATBOT
        # -------------------------
        with right:

            st.markdown("### Ask Questions About This Report")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_question = st.text_input("Enter your question")

            if st.button("Ask") and user_question:

                retrieved = memory.retrieve(user_question, k=5)
                context = "\n\n".join(retrieved)

                if context:
                    answer = call_llm(
                        "Answer strictly using the provided context. If the information is not present, clearly state it is not available in the report.",
                        f"Context:\n{context}\n\nQuestion:\n{user_question}",
                        0.3
                    )
                else:
                    answer = "No relevant information found in the report."

                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("assistant", answer))

            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**Assistant:** {message}")


# ==========================================================
# OTHER PAGES
# ==========================================================

elif page == "Architecture":
    st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)

    st.markdown("### High-Level Architecture Diagram")

    st.image(
        "architecture_diagram.png",
        use_container_width=True
    )


elif page == "Why Not ChatGPT?":
    st.markdown('<div class="section-title">Comparative Positioning</div>', unsafe_allow_html=True)

    st.markdown("""
### 1. Architectural Transparency
This system exposes planning, retrieval, critique, and refinement stages.  
Generic AI systems operate as black boxes.  
Here, each stage is interpretable.  
This improves academic defensibility.

### 2. Grounded Knowledge Retrieval
The system performs live web retrieval.  
ChatGPT often relies on pretrained knowledge.  
Grounded retrieval ensures updated information.  
It reduces hallucination probability.

### 3. Multi-Agent Reasoning
Instead of a single-pass response, this system employs multiple reasoning agents.  
Planning and critique simulate structured cognition.  
It mimics research workflows.  
This increases analytical depth.

### 4. Reflective Critique Mechanism
The Critic Agent evaluates logical gaps.  
Generic LLM responses lack internal review loops.  
Reflective critique improves quality.  
It simulates peer-review standards.

### 5. Retrieval-Augmented Context Control
All outputs are grounded in verified memory chunks.  
This ensures contextual consistency.  
Generic AI tools may introduce unrelated information.  
Context grounding enhances reliability.

### 6. Academic Orientation
The system enforces structured headings and analytical format.  
It is optimized for research reports.  
Generic assistants prioritize conversational output.  
This system prioritizes structured scholarship.

### 7. Reduced Hallucination Risk
Since generation depends on retrieved embeddings, fabrication is minimized.  
Hallucination risk is structurally reduced.  
This makes it suitable for academic use.  
Reliability is improved.

### 8. Evaluation Visibility
Intermediate outputs such as draft and critique can be inspected.  
Users understand how refinement occurs.  
Generic systems hide intermediate reasoning.  
Transparency strengthens trust.

### 9. Customizable Pipeline
Model selection, retrieval depth, and refinement steps are configurable.  
Generic systems offer limited control.  
This architecture is adaptable.  
It supports experimentation.

### 10. Research-Oriented Interaction
The Assistant interacts strictly within report context.  
It functions as a research companion.  
Generic systems provide broad answers.  
This system maintains topical discipline.
""")



elif page == "Evaluation Strategy":
    st.markdown('<div class="section-title">Evaluation Strategy</div>', unsafe_allow_html=True)

    st.markdown("""
### 1. Groundedness Verification
Evaluate whether generated statements align with retrieved context.  
Cross-check claims against stored embeddings.  
Ensure no unsupported assertions exist.  
This measures factual reliability.

### 2. Retrieval Precision
Measure relevance of retrieved chunks for each query.  
Assess semantic similarity alignment.  
Evaluate redundancy reduction.  
Precision indicates retrieval quality.

### 3. Hallucination Reduction
Compare draft with retrieved context.  
Identify unsupported expansions.  
Critique loop should reduce hallucination rate.  
Improvement validates architecture.

### 4. Structural Coherence
Assess logical flow of sections.  
Evaluate clarity of argument transitions.  
Ensure heading hierarchy consistency.  
This validates academic formatting.

### 5. Comparative Benchmarking
Compare output with baseline LLM single-pass generation.  
Evaluate analytical depth differences.  
Measure structural completeness.  
Quantify improvement over generic responses.

### 6. User Interaction Testing
Evaluate assistant responses to follow-up queries.  
Ensure contextual grounding.  
Test memory retrieval consistency.  
Assess conversational reliability.
""")



elif page == "Future Scope":
    st.markdown('<div class="section-title">Future Scope</div>', unsafe_allow_html=True)

    st.markdown("""
### 1. Academic Database Integration
Integrate APIs such as ArXiv, PubMed, or IEEE.  
Enable peer-reviewed paper retrieval.  
Improve academic credibility.  
Enhance citation quality.

### 2. Hybrid Retrieval Models
Combine BM25 keyword retrieval with embedding search.  
Improve recall for technical queries.  
Enhance ranking robustness.  
Reduce semantic drift.

### 3. Persistent Vector Storage
Replace session-based FAISS with persistent storage.  
Enable long-term research memory.  
Support multi-session continuity.  
Scale system for enterprise use.

### 4. Multi-Critic Debate Agents
Introduce adversarial critic agents.  
Allow structured argument counter-analysis.  
Simulate academic debate.  
Improve analytical rigor.

### 5. Citation Generator Module
Automatically format references in APA/IEEE style.  
Extract source URLs and titles.  
Ensure traceability.  
Improve academic usability.

### 6. Research Visualization Layer
Generate research maps or topic graphs.  
Visualize focus areas.  
Display retrieval clusters.  
Enhance interpretability.
""")



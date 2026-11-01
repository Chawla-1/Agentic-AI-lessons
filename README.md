# 🤖 AI Agent Engineering from First Principles

A comprehensive, ground-up implementation of autonomous AI agent architectures using local LLMs (**Qwen3**) and vector similarity search (**FAISS**).

This repository traces the complete cognitive evolution of an AI agent—from a primitive rule-based function dispatcher to an autonomous, self-healing agent capable of dependency-aware DAG planning, dynamic replanning, and full memory lifecycle management.

---

## 🏛️ Architecture Progression (V01 → V12)

```
┌────────────────────────────────────────────────────────────────────────┐
│                      AGENT COGNITIVE MILESTONES                        │
├────────────────────────────────────────────────────────────────────────┤
│ v01: Simulated Agent & Tool Calling (Rule-based decision tree)         │
│ v02: Natural Language Argument Parser (Simulated parameter extraction) │
│ v03: ReAct Execution Engine (Observation-action feedback loop)         │
│ v04: Dynamic Tool Registry (JSON Schema validation & dict dispatch)    │
│ v05: Local LLM Integration (Qwen3 native function calling via Ollama)  │
│ v06: Stateful Conversation & Multi-Step Planning (History tracking)   │
│ v07: Retrieval-Augmented Generation (Dense embeddings + FAISS index)   │
│ v08: Reflection & Error Recovery (Math router + self-verification)     │
│ v09: Structured Persistent Memory (Atomic writes + sliding window)     │
│ v10: Semantic Vector Long-Term Memory (Cosine similarity filtering)    │
│ v11: Memory Lifecycle (Conflict resolution, TTL expiry & provenance)   │
│ v12: Reliable Planning & Recovery (DAG dependencies & dynamic replan)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

All agent source files reside in the [`ai-agent/`](./ai-agent/) directory:

| Version | File | Core Concept Demonstrated |
|---|---|---|
| **V01** | [`v01_simulated_tools.py`](./ai-agent/v01_simulated_tools.py) | **Foundation:** Simulated LLM decision-making, tool execution mapping, and basic chatbot logic. |
| **V02** | [`v02_argument_extraction.py`](./ai-agent/v02_argument_extraction.py) | **Parameter Parsing:** Extracting structured tool arguments from unstructured natural language queries. |
| **V03** | [`v03_execution_loop.py`](./ai-agent/v03_execution_loop.py) | **ReAct Loop:** Autonomous observation-action cycle with `MAX_ITERATIONS` guardrails to prevent infinite loops. |
| **V04** | [`v04_tool_registry.py`](./ai-agent/v04_tool_registry.py) | **Modularity:** Tool schemas defined in OpenAI-compatible JSON function format with clean dictionary dispatching. |
| **V05** | [`v05_local_llm_qwen3.py`](./ai-agent/v05_local_llm_qwen3.py) | **Local Inference:** Real local LLM integration using `ollama` with `qwen3` handling tool selection and execution. |
| **V06** | [`v06_conversation_and_planning.py`](./ai-agent/v06_conversation_and_planning.py) | **Context & Planning:** Maintaining multi-turn conversation memory and executing multi-step goals sequentially. |
| **V07** | [`v07_rag_faiss_vector_store.py`](./ai-agent/v07_rag_faiss_vector_store.py) | **RAG Engine:** Document chunking, dense semantic embedding (`nomic-embed-text`), and persistent FAISS indexing. |
| **V08** | [`v08_reflection_and_verification.py`](./ai-agent/v08_reflection_and_verification.py) | **Self-Correction:** Deterministic fast-path routing, AST safe evaluation, and post-action result verification with retries. |
| **V09** | [`v09_structured_persistent_memory.py`](./ai-agent/v09_structured_persistent_memory.py) | **Persistent Profile:** Atomic disk persistence (`.tmp` -> `replace`), LLM fact extraction, deduplication, and sliding-window history. |
| **V10** | [`v10_semantic_vector_memory.py`](./ai-agent/v10_semantic_vector_memory.py) | **Semantic Memory:** Atomic fact vector indexing with strict cosine similarity cutoff ($\tau = 0.45$) to eliminate context window pollution. |
| **V11** | [`v11_memory_lifecycle.py`](./ai-agent/v11_memory_lifecycle.py) | **Memory Lifecycle:** Semantic conflict resolution (LLM judge), automated TTL expiry sweeps, provenance tracking, and surgical forgetting (`/forget`). |
| **V12** | [`v12_reliable_planning.py`](./ai-agent/v12_reliable_planning.py) | **Deterministic Planning:** Goal decomposition into Directed Acyclic Graphs (DAGs), precondition enforcement, Devin-style live checklists, and autonomous dynamic replanning. |

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** running locally with required models:
   ```bash
   ollama pull qwen3
   ollama pull nomic-embed-text
   ```

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Chawla-1/Agentic-AI-lessons.git
cd Agentic-AI-lessons
pip install ollama faiss-cpu numpy
```

### Running an Agent

Run any milestone directly. For example, to run the definitive planning agent:

```bash
cd ai-agent
python v12_reliable_planning.py
```

#### Available Commands in V12:
- `/plan <goal>` — Explicitly decompose and execute a multi-step mission with a live status checklist.
- `/memory` — Inspect all active atomic facts and TTL expiration timers.
- `/memory_all` — Inspect the full lifecycle audit ledger (including superseded and expired records).
- `/clear_history` — Reset the current session's short-term conversation context.
- `/clear_memory` — Wipe the persistent vector memory index.
- `exit` — Terminate session.

---

## 🛠️ Key Technical Highlights

- **Autonomous Dynamic Replanning:** When an intermediate step fails (e.g. division by zero or empty search results), the engine catches the error, halts forward execution, prompts the LLM with the failure traceback, and autonomously rewrites the remaining DAG to recover.
- **Semantic Conflict Judge:** Solves the mutability problem in vector stores. When a new candidate fact matches an existing fact ($\ge 0.60$ cosine similarity), an LLM judge decides whether to supersede/replace the old fact or keep both.
- **Zero-Waste Context Injection:** The semantic memory engine computes cosine similarity between incoming queries and memory facts, completely withholding memory tokens when the user asks irrelevant questions.
- **Fail-Safe Persistence:** Uses OS-level atomic file replacement (`os.replace`) to guard against state corruption during unexpected crashes or power interruptions.
- **Sandboxed Evaluation:** Uses Python's `ast.parse` and abstract syntax tree traversal to compute math expressions safely without vulnerable `eval()`.
- **Local & Private:** Zero API costs, zero telemetry, and 100% offline-capable execution.

---

## 🗺️ Roadmap & Upcoming Milestones

- [x] **Phase 0:** Foundations (Tool Calling, Loops, Local Qwen3, RAG, Reflection)
- [x] **Phase 1 (Milestone 1–2):** Deterministic Memory & Semantic Vector Long-Term Memory
- [x] **Phase 1 (Milestone 3):** Memory Lifecycle (Conflict resolution, TTL expiry, provenance)
- [x] **Phase 1 (Milestone 4):** Reliable Planning & Recovery (DAG dependency execution, replanning)
- [ ] **Phase 1 (Milestone 5):** Structured Outputs & Schema Enforcement (Pydantic validation, malformed output recovery)
- [ ] **Phase 1 (Milestone 6):** Agent Security Fundamentals (Prompt injection defense, untrusted content sanitization, approval gates)
- [ ] **Phase 2:** Advanced Execution (Parallel tool execution via `asyncio`, Multi-agent swarms)
- [ ] **Phase 3:** Production Frameworks (LangGraph, custom checkpointers, human-in-the-loop)

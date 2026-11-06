# 🤖 AI Agent Engineering from First Principles

A comprehensive, ground-up implementation of autonomous AI agent architectures using local LLMs (**Qwen3**) and vector similarity search (**FAISS**).

This repository traces the complete cognitive evolution of an AI agent—from a primitive rule-based function dispatcher to an autonomous, self-healing agent capable of dependency-aware DAG planning, dynamic replanning, multi-worker parallel execution, full memory lifecycle management, and hierarchical multi-agent architectures.

---

## 🏛️ Architecture Progression (V01 → V16)

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
│ v13: Structured Outputs & Validation (Pydantic contracts & self-repair)│
│ v14: Agent Security Fundamentals (Least privilege, gates & sandboxing) │
│ v15: Parallel Execution & Wavefront Scheduler (Concurrent DAG executor)│
│ v16: Multi-Agent Architecture (Supervisor, specialists & blackboard)  │
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
| **V13** | [`v13_structured_outputs.py`](./ai-agent/v13_structured_outputs.py) | **Structured Outputs & Schema Contracts:** Pydantic domain models, type-safe argument parsing, and automated schema repair loops for malformed LLM outputs. |
| **V14** | [`v14_agent_security.py`](./ai-agent/v14_agent_security.py) | **Agent Security Architecture:** Defense-in-depth, prompt injection defense, untrusted context taint tracking, least-privilege RBAC tiers, human approval gates, and circuit breakers. |
| **V15** | [`v15_parallel_execution.py`](./ai-agent/v15_parallel_execution.py) | **Parallel Execution & Wavefront Scheduler:** Concurrent tool execution using `ThreadPoolExecutor`, topological wavefront scheduling, timeout isolation, and generalized intent-coverage synthesis. |
| **V16** | [`v16_multi_agent.py`](./ai-agent/v16_multi_agent.py) | **Hierarchical Multi-Agent Architecture:** Supervisor orchestrator, plan-first task decomposition, specialist agents (`ResearcherAgent`, `ComputationAgent`, `CriticAgent`), shared Blackboard state, and adversarial verification loops. |

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
pip install ollama faiss-cpu numpy pydantic
```

### Running an Agent

Run any milestone directly. For example, to run the multi-agent system:

```bash
cd ai-agent
python v16_multi_agent.py
```

#### Available Commands in V16:
- `/blackboard` — Inspect active blackboard state, artifacts, and task history.
- `/memory` — Inspect active atomic biographical facts from semantic vector memory.
- `exit` — Terminate session.

---

## 🛠️ Key Technical Highlights

- **Hierarchical Multi-Agent Orchestration:** Decomposes complex missions into specialized domains. A central **SupervisorAgent** delegates tasks to specialist workers (`ResearcherAgent`, `ComputationAgent`, `CriticAgent`) via typed Pydantic contracts.
- **Shared Blackboard Architecture:** Decouples worker agents through a thread-safe `BlackboardState` storing verified artifacts, user context, and execution audit history without context window contamination.
- **Adversarial Critic & Disagreement Protocol:** An independent `CriticAgent` audits intermediate deliverables against the original user goal before conclusion. Discrepancies trigger targeted revision loops to correct errors before final synthesis.
- **Concurrent Wavefront DAG Scheduler (V15):** Groups zero-dependency tasks into parallel wavefronts via `ThreadPoolExecutor` with barrier synchronization.
- **Automated Pydantic Schema Self-Repair:** Intercepts JSON parsing and validation failures, prompts Qwen3 with schema diffs, and heals malformed payloads automatically.
- **Defense-in-Depth Security Gateways:** Enforces prompt injection defense, sandboxed AST math evaluation, and approval gates for destructive operations.
- **Local & Private:** Zero API costs, zero telemetry, and 100% offline-capable execution.

---

## 🗺️ Roadmap Progress

- [x] **Phase 0:** Foundations (Tool Calling, Loops, Local Qwen3, RAG, Reflection)
- [x] **Phase 1 (Milestone 1–2):** Deterministic Memory & Semantic Vector Long-Term Memory
- [x] **Phase 1 (Milestone 3):** Memory Lifecycle (Conflict resolution, TTL expiry, provenance)
- [x] **Phase 1 (Milestone 4):** Reliable Planning & Recovery (DAG dependency execution, replanning)
- [x] **Phase 1 (Milestone 5):** Structured Outputs & Schema Enforcement (`v13_structured_outputs.py`)
- [x] **Phase 1 (Milestone 6):** Agent Security Fundamentals (`v14_agent_security.py`)
- [x] **Phase 2 (Milestone 1):** Parallel Tool Execution & Wavefront Scheduler (`v15_parallel_execution.py`)
- [x] **Phase 2 (Milestone 2):** Multi-Agent Architecture (`v16_multi_agent.py`)
- [ ] **Phase 2 (Milestone 3):** Human-in-the-Loop Workflows
- [ ] **Phase 2 (Milestone 4):** Evaluation & Observability
- [ ] **Phase 3:** Production Frameworks (LangGraph, custom checkpointers, stateful graphs)

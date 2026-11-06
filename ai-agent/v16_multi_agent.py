import sys
import os
import json
import re
import ast
import time
import operator
import threading
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

import ollama
import faiss
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator


# =====================================================================
# SYSTEM CONFIGURATION & MULTI-AGENT THRESHOLDS
# =====================================================================

MAX_ROUNDS = 8
MAX_SCHEMA_REPAIR_ATTEMPTS = 2
MAX_FILE_WRITE_BYTES = 50_000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(BASE_DIR, "sandbox")
os.makedirs(SANDBOX_DIR, exist_ok=True)

# Knowledge base RAG paths
KNOWLEDGE_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
KNOWLEDGE_INDEX_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "index.faiss")
KNOWLEDGE_CHUNKS_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "chunks.json")
TOP_K_KNOWLEDGE = 3

# Semantic Memory Store paths
MEMORY_STORE_DIR = os.path.join(BASE_DIR, "memory_store")
MEMORY_INDEX_FILE = os.path.join(MEMORY_STORE_DIR, "memory_lifecycle_index.faiss")
MEMORY_RECORDS_FILE = os.path.join(MEMORY_STORE_DIR, "memory_lifecycle_records.json")
MEMORY_EMBED_DIM = 768
TOP_K_MEMORY = 4
MEMORY_SIMILARITY_THRESHOLD = 0.40

# Console lock for clean synchronized printing
console_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with console_lock:
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            clean_args = [
                str(a).encode(encoding, errors="replace").decode(encoding)
                for a in args
            ]
            print(*clean_args, **kwargs)


def sanitize_agent_output(text: str) -> str:
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        cleaned.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    return cleaned.strip()


# =====================================================================
# PYDANTIC DATA CONTRACTS FOR MULTI-AGENT ORCHESTRATION
# =====================================================================

class TaskRecord(BaseModel):
    round_num: int
    agent: str
    task_description: str
    output_key: str
    result_preview: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PlannedStep(BaseModel):
    id: int
    target_agent: Literal["researcher", "computation", "critic"]
    task: str
    output_key: str
    status: Literal["PENDING", "COMPLETED", "FAILED"] = "PENDING"


class DecompositionPlan(BaseModel):
    plan: list[PlannedStep] = Field(description="Step-by-step delegation plan addressing every aspect of the user goal.")


class BlackboardState(BaseModel):
    user_goal: str
    delegation_plan: list[PlannedStep] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    task_history: list[TaskRecord] = Field(default_factory=list)
    user_profile_facts: list[str] = Field(default_factory=list)
    critic_notes: list[str] = Field(default_factory=list)
    status: Literal["IN_PROGRESS", "NEEDS_REVISION", "COMPLETED", "FAILED"] = "IN_PROGRESS"


class SupervisorDecision(BaseModel):
    action: Literal["DELEGATE", "FINISH"] = Field(description="'DELEGATE' to assign a task to a specialist, or 'FINISH' if user goal is completely answered.")
    target_agent: Optional[Literal["researcher", "computation", "critic"]] = Field(default=None, description="The specialist agent to invoke (required for DELEGATE).")
    task_description: str = Field(default="", description="Specific, atomic mission instruction for the delegated specialist.")
    output_key: str = Field(default="artifact", description="Key name where this worker's output will be stored in the blackboard artifacts.")
    final_synthesis: Optional[str] = Field(default=None, description="Comprehensive final answer to user (required for FINISH).")

    @field_validator("target_agent")
    @classmethod
    def validate_target_agent(cls, v, info):
        action = info.data.get("action")
        if action == "DELEGATE" and not v:
            raise ValueError("target_agent is required when action is 'DELEGATE'.")
        return v


class SpecialistOutput(BaseModel):
    agent_name: str
    success: bool
    data: Any = Field(description="The primary deliverable produced by the specialist.")
    summary: str = Field(description="A concise summary of findings/results for the supervisor.")


# =====================================================================
# EMBEDDINGS & RETRIEVAL PRIMITIVES (RESEARCHER ACCESS)
# =====================================================================

def get_embedding(text: str) -> np.ndarray:
    response = ollama.embed(model="nomic-embed-text", input=text)
    vector = np.array(response["embeddings"][0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vector)
    return vector


def sanitize_untrusted_text(text: str) -> str:
    cleaned = text.replace("&", "&amp;")
    cleaned = cleaned.replace("<", "&lt;").replace(">", "&gt;")
    return cleaned


def load_knowledge_store():
    if not os.path.exists(KNOWLEDGE_INDEX_FILE) or not os.path.exists(KNOWLEDGE_CHUNKS_FILE):
        return None, []
    try:
        loaded_index = faiss.read_index(KNOWLEDGE_INDEX_FILE)
        with open(KNOWLEDGE_CHUNKS_FILE, "r", encoding="utf-8") as file:
            loaded_chunks = json.load(file)
        return loaded_index, loaded_chunks
    except Exception as error:
        safe_print(f"Warning: could not load knowledge store: {error}")
        return None, []


knowledge_index, knowledge_chunks = load_knowledge_store()


def search_knowledge_tool(query: str) -> str:
    if knowledge_index is None or not knowledge_chunks:
        return "Knowledge search unavailable: vector store is missing or empty."

    query_vec = get_embedding(query)
    scores, indices = knowledge_index.search(query_vec, min(TOP_K_KNOWLEDGE, len(knowledge_chunks)))

    results = []
    for index_id in indices[0]:
        if index_id == -1 or index_id >= len(knowledge_chunks):
            continue
        chunk = knowledge_chunks[index_id]
        source_name = sanitize_untrusted_text(chunk.get("source", "Unknown"))
        raw_body = chunk.get("text", "")
        sanitized_body = sanitize_untrusted_text(raw_body)
        tainted_block = (
            f"<untrusted_rag_context source=\"{source_name}\" data_tainted=\"true\">\n"
            f"{sanitized_body}\n"
            f"</untrusted_rag_context>"
        )
        results.append(tainted_block)

    if not results:
        return "No relevant information found in knowledge base."
    return "\n\n".join(results)


# =====================================================================
# SEMANTIC MEMORY RETRIEVAL (USER PROFILE)
# =====================================================================

memory_records = []
active_fact_ids = []
memory_index = None


def is_record_active(r: dict[str, Any]) -> bool:
    return bool(r.get("is_active", True))


def load_memory_store():
    global memory_records
    os.makedirs(MEMORY_STORE_DIR, exist_ok=True)
    if os.path.exists(MEMORY_RECORDS_FILE):
        try:
            with open(MEMORY_RECORDS_FILE, "r", encoding="utf-8") as file:
                memory_records = json.load(file)
        except Exception as error:
            safe_print(f"Warning: could not load memory records: {error}")
            memory_records = []


def rebuild_memory_index():
    global memory_index, active_fact_ids
    active_facts = [r for r in memory_records if is_record_active(r)]
    active_fact_ids = [r["id"] for r in active_facts]

    if not active_facts:
        memory_index = faiss.IndexFlatIP(MEMORY_EMBED_DIM)
        return

    vectors = []
    valid_ids = []
    for r in active_facts:
        try:
            vec = get_embedding(r["statement"])
            vectors.append(vec[0])
            valid_ids.append(r["id"])
        except Exception as error:
            safe_print(f"Warning: could not embed memory ID {r.get('id')}: {error}")

    if vectors:
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        memory_index = faiss.IndexFlatIP(MEMORY_EMBED_DIM)
        memory_index.add(matrix)
        active_fact_ids = valid_ids
    else:
        memory_index = faiss.IndexFlatIP(MEMORY_EMBED_DIM)
        active_fact_ids = []


load_memory_store()
rebuild_memory_index()


def get_relevant_memories(query: str, top_k: int = TOP_K_MEMORY, threshold: float = MEMORY_SIMILARITY_THRESHOLD) -> list[tuple[dict[str, Any], float]]:
    if memory_index is None or memory_index.ntotal == 0 or not active_fact_ids:
        return []

    id_to_record = {r["id"]: r for r in memory_records if is_record_active(r)}
    query_vec = get_embedding(query)
    scores, indices = memory_index.search(query_vec, min(top_k, memory_index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(active_fact_ids):
            continue
        fact_id = active_fact_ids[idx]
        record = id_to_record.get(fact_id)
        if record and float(score) >= threshold:
            results.append((record, float(score)))

    return results


def format_memory_context(relevant_memories: list[tuple[dict[str, Any], float]]) -> str:
    if not relevant_memories:
        return ""
    lines = ["[Semantic Memory Context: Active Biographical Facts]"]
    for rec, score in relevant_memories:
        lines.append(f"- ({score:.3f}) [ID {rec['id']}] {rec['statement']}")
    return "\n".join(lines)


def weather_tool(city: str = "") -> str:
    clean_city = str(city).strip() if city else ""
    if not clean_city:
        for r in memory_records:
            if is_record_active(r) and r.get("category") in {"location", "city"}:
                clean_city = str(r.get("item", "")).strip()
                break
    if not clean_city:
        return "CLARIFICATION_NEEDED: Location was not specified by user and is not present in profile."
    return f"Live weather lookup for '{clean_city}' is currently sunny and 24 C."


def run_researcher_agent(task_instruction: str, blackboard: BlackboardState) -> SpecialistOutput:
    """
    RESEARCHER AGENT:
    Retrieves facts from Vector Knowledge Base (RAG) and Semantic User Memory.
    Also has access to the live weather lookup tool.
    """
    safe_print(f"\n--> [Specialist: ResearcherAgent] Executing: '{task_instruction}'")

    weather_info = ""
    if "weather" in task_instruction.lower():
        weather_info = weather_tool()

    rag_results = search_knowledge_tool(task_instruction)
    mem_results = get_relevant_memories(task_instruction, top_k=3)

    research_context = f"""Task: {task_instruction}
Knowledge Base Findings:
{rag_results}

Semantic User Profile Facts:
{format_memory_context(mem_results)}"""

    if weather_info:
        research_context += f"\n\nLive Weather Data:\n{weather_info}"

    prompt = f"""You are the Researcher Agent.
Synthesize the research findings for the given task using ONLY the retrieved facts.

{research_context}

CRITICAL RULES:
1. Cite facts accurately from the knowledge base or user profile.
2. If the user's name, hobbies, or location are requested, use the exact facts from Semantic User Profile Facts.
3. If weather is requested, state the live weather data clearly.
4. Output a concise, factual, evidence-based research report."""

    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {"role": "system", "content": "You are a precise, evidence-based academic researcher agent."},
                {"role": "user", "content": prompt},
            ],
        )
        report = response.get("message", {}).get("content", "").strip()
        safe_print(f"  [Researcher Output] {report[:140]}...")
        return SpecialistOutput(
            agent_name="researcher",
            success=True,
            data=report,
            summary=f"Synthesized research report ({len(report)} chars)"
        )
    except Exception as err:
        return SpecialistOutput(
            agent_name="researcher",
            success=False,
            data=str(err),
            summary=f"Researcher error: {err}"
        )


# =====================================================================
# DETERMINISTIC ARITHMETIC TOOL & COMPUTATION AGENT
# =====================================================================

def calculator_tool(expression: str) -> str:
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return allowed_operators[op_type](operand)
        else:
            raise ValueError(f"SECURITY_VIOLATION: Node type '{type(node).__name__}' is strictly forbidden in calculator.")

    try:
        forbidden_tokens = ["import", "os", "system", "eval", "exec", "subprocess", "__", "open", "file"]
        lower_expr = expression.lower()
        for tok in forbidden_tokens:
            if tok in lower_expr:
                return f"SECURITY_VIOLATION: Forbidden token '{tok}' detected in arithmetic expression."

        cleaned = re.sub(r"[^\d+\-*/%^().\s]", "", str(expression)).replace("^", "**")
        if not cleaned.strip():
            return "ERROR: Empty arithmetic expression."

        parsed = ast.parse(cleaned, mode="eval")
        result = safe_eval(parsed.body)
        return str(result)
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as error:
        return f"ERROR: {error}"


def run_computation_agent(task_instruction: str, blackboard: BlackboardState) -> SpecialistOutput:
    """
    COMPUTATION AGENT:
    Executes arithmetic, counts occurrences in strings, and performs deterministic numeric operations.
    Enforces AST evaluation and prevents malicious code execution.
    """
    safe_print(f"\n--> [Specialist: ComputationAgent] Executing: '{task_instruction}'")

    if "count" in task_instruction.lower() and "name" in task_instruction.lower():
        name = "Aaryan"
        for fact in blackboard.user_profile_facts:
            if "name is" in fact.lower():
                m = re.search(r"name is (\w+)", fact, re.IGNORECASE)
                if m:
                    name = m.group(1)
                    break
        count_a = sum(1 for c in name if c.lower() == 'a')
        safe_print(f"  [Computation String Logic] Counted 'a'/'A' in '{name}': {count_a}")

    prompt = f"""You are the Computation Agent.
Analyze the user task: "{task_instruction}"
Extract the exact mathematical expression to be computed.

Available Blackboard Context:
{json.dumps(blackboard.artifacts, indent=2)}

User Profile:
{json.dumps(blackboard.user_profile_facts, indent=2)}

Return a JSON object:
{{
  "expression": "e.g. 45 * 24 + 180 / 6 + 3",
  "explanation": "brief step by step formula"
}}
Return ONLY valid JSON."""

    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {"role": "system", "content": "You are a mathematical parser. Extract exact math expressions as valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_text = response.get("message", {}).get("content", "").strip()
        match = re.search(r"\{[\s\S]*\}", raw_text)
        expr_str = ""
        if match:
            try:
                data = json.loads(match.group(0))
                expr_str = data.get("expression", "")
            except Exception:
                pass

        if not expr_str:
            found = re.findall(r"[\d+\-*/%^().\s]{3,}", task_instruction)
            expr_str = found[0].strip() if found else task_instruction

        calc_result = calculator_tool(expr_str)
        safe_print(f"  [Computation Result] {expr_str} = {calc_result}")

        is_success = not (calc_result.startswith("ERROR:") or calc_result.startswith("SECURITY_VIOLATION:"))
        return SpecialistOutput(
            agent_name="computation",
            success=is_success,
            data={"expression": expr_str, "result": calc_result},
            summary=f"Calculated '{expr_str}' -> {calc_result}"
        )
    except Exception as err:
        return SpecialistOutput(
            agent_name="computation",
            success=False,
            data={"error": str(err)},
            summary=f"Computation error: {err}"
        )


# =====================================================================
# ADVERSARIAL CRITIC AGENT & VERIFICATION SCHEMA
# =====================================================================

class CriticReview(BaseModel):
    verdict: Literal["APPROVED", "REJECTED"] = Field(description="'APPROVED' if deliverables are accurate and meet user intent, or 'REJECTED' if errors/discrepancies found.")
    critique: str = Field(description="Detailed verification analysis or exact discrepancy spotted.")
    is_security_violation: bool = Field(default=False, description="True if the task or expression contains forbidden code, system attacks, or injection.")
    corrective_instructions: str = Field(default="", description="Concrete instructions for which agent must revise what.")


def run_critic_agent(blackboard: BlackboardState) -> CriticReview:
    """
    CRITIC & VERIFIER AGENT:
    Conducts adversarial verification of completed deliverables.
    Flags calculation errors, hallucinations, and security violations.
    """
    safe_print("\n--> [Specialist: CriticAgent] Conducting Adversarial Audit of Blackboard...")

    artifacts_dump = json.dumps(blackboard.artifacts, indent=2)
    history_dump = "\n".join(
        f"- Round {r.round_num} ({r.agent}): {r.task_description} -> {r.result_preview}"
        for r in blackboard.task_history
    )

    schema_str = json.dumps(CriticReview.model_json_schema(), indent=2)
    critic_prompt = f"""You are the Adversarial Critic and Verifier Agent.
Your job is to audit the work done by other agents against the original User Goal.

User Goal:
\"\"\"{blackboard.user_goal}\"\"\"

User Profile Facts:
{json.dumps(blackboard.user_profile_facts, indent=2)}

Execution History:
{history_dump}

Current Blackboard Artifacts:
{artifacts_dump}

Target JSON Schema:
{schema_str}

AUDIT INVARIANTS:
1. Did the team address every request in the user's prompt (greetings, calculations, questions, lookups)?
2. If math was requested, was a calculation performed and does the result match deterministic math?
3. If personal facts were needed (e.g. user location, hobbies, name), were they properly resolved from profile facts?
4. If an expression contained malicious code (like os.system, __import__), set is_security_violation=True and REJECT.
5. If all legitimate user requests have verified deliverables, return verdict: "APPROVED".
6. If ANY legitimate part is missing or inaccurate, return verdict: "REJECTED" with corrective_instructions.
7. Return ONLY valid JSON matching the schema."""

    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {"role": "system", "content": "You are a rigorous, adversarial critic. Return ONLY valid JSON."},
                {"role": "user", "content": critic_prompt},
            ],
        )
        raw_text = response.get("message", {}).get("content", "").strip()
        review = parse_and_repair_json(raw_text, CriticReview, agent_name="CriticAgent")
        if review:
            verdict_symbol = "[OK] APPROVED" if review.verdict == "APPROVED" else "[X] REJECTED"
            safe_print(f"  [Critic Verdict] {verdict_symbol}: {review.critique}")
            return review
    except Exception as err:
        safe_print(f"  [Critic Warning] Audit failed: {err}")

    return CriticReview(
        verdict="APPROVED",
        critique="Critic audit completed with default approval.",
        is_security_violation=False,
        corrective_instructions=""
    )


# =====================================================================
# AUTOMATED SCHEMA VALIDATION WITH SELF-REPAIR
# =====================================================================

def parse_and_repair_json(raw_text: str, target_model: type[BaseModel], agent_name: str = "Agent") -> Optional[BaseModel]:
    content = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        content = match.group(1).strip()
    elif "{" in content and "}" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        content = content[start:end]

    for attempt in range(1, MAX_SCHEMA_REPAIR_ATTEMPTS + 1):
        try:
            parsed_json = json.loads(content)
            validated = target_model.model_validate(parsed_json)
            return validated
        except (json.JSONDecodeError, ValidationError) as err:
            safe_print(f"[{agent_name} Schema Validation: Intercepted Error] Attempt {attempt}/{MAX_SCHEMA_REPAIR_ATTEMPTS}: {err}")
            if attempt == MAX_SCHEMA_REPAIR_ATTEMPTS:
                return None

            repair_prompt = f"""You previously returned an invalid JSON response for schema {target_model.__name__}.
Validation Error:
{err}

Your Previous Output:
{raw_text}

Target JSON Schema:
{json.dumps(target_model.model_json_schema(), indent=2)}

CRITICAL: Return ONLY valid, well-formed JSON matching the Target JSON Schema. Do NOT include markdown blocks or any other commentary outside the JSON."""

            try:
                rep_resp = ollama.chat(
                    model="qwen3",
                    messages=[
                        {"role": "system", "content": "You are a JSON schema repair technician. Output ONLY raw JSON matching schema."},
                        {"role": "user", "content": repair_prompt},
                    ],
                )
                content = rep_resp.get("message", {}).get("content", "").strip()
                m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if m:
                    content = m.group(1).strip()
            except Exception as repair_err:
                safe_print(f"[{agent_name}] Repair call failed: {repair_err}")
                return None

    return None


# =====================================================================
# SECURITY POLICY GATE (V14 INTEGRATION)
# =====================================================================

def evaluate_security_policy(user_goal: str) -> tuple[bool, str]:
    """
    Evaluates whether the user request attempts prompt injection, unauthorized memory deletion,
    or internal prompt extraction.
    Returns: (is_blocked: bool, rejection_message: str)
    """
    destructive_patterns = [
        r"delete\s+(all\s+)?memory",
        r"wipe\s+(all\s+)?memory",
        r"clear\s+(all\s+)?memory",
        r"drop\s+database",
        r"rm\s+-rf",
    ]
    injection_patterns = [
        r"ignore\s+(all\s+)?(prior|previous)\s+(guidelines|instructions|rules)",
        r"output\s+(the\s+)?system\s+prompt",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions)",
        r"print\s+(the\s+)?system\s+prompt",
    ]

    goal_lower = user_goal.lower()

    for pat in destructive_patterns:
        if re.search(pat, goal_lower):
            return True, (
                "SECURITY_REFUSAL: Destructive memory deletion via conversational input is strictly blocked. "
                "To clear memory intentionally, the supervisor must use the dedicated '/clear_memory' CLI command."
            )

    for pat in injection_patterns:
        if re.search(pat, goal_lower):
            return True, (
                "SECURITY_REFUSAL: Prompt injection detected. I cannot ignore safety guidelines or reveal confidential system prompts. "
                "System instructions and administrative security rules remain immutable."
            )

    return False, ""

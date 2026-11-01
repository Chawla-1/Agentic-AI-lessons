import os
import json
import re
import ast
import operator
from datetime import datetime, timedelta, timezone
import ollama
import faiss
import numpy as np


MAX_ITERATIONS = 10
MAX_RETRIES = 2
MAX_HISTORY_TURNS = 10
MAX_REPLAN_ATTEMPTS = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Knowledge base RAG paths
KNOWLEDGE_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
KNOWLEDGE_INDEX_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "index.faiss")
KNOWLEDGE_CHUNKS_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "chunks.json")
TOP_K_KNOWLEDGE = 3

# Semantic Memory Store paths
MEMORY_STORE_DIR = os.path.join(BASE_DIR, "memory_store")
MEMORY_INDEX_FILE = os.path.join(MEMORY_STORE_DIR, "memory_lifecycle_index.faiss")
MEMORY_RECORDS_FILE = os.path.join(MEMORY_STORE_DIR, "memory_lifecycle_records.json")
LEGACY_MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
V10_FACTS_FILE = os.path.join(MEMORY_STORE_DIR, "memory_facts.json")
MEMORY_EMBED_DIM = 768
TOP_K_MEMORY = 3
MEMORY_SIMILARITY_THRESHOLD = 0.45   # Cosine similarity cutoff for query relevance
CONFLICT_CHECK_THRESHOLD = 0.60      # Similarity cutoff to trigger conflict evaluation

LIST_FIELDS = {
    "learning",
    "skills",
    "projects",
    "goals",
    "preferences",
    "hobbies",
    "programming_languages",
    "tools",
}

MEMORY_LABELS = {
    "name": "name",
    "age": "age",
    "education": "education",
    "location": "location / city",
    "learning": "learning",
    "skills": "skills",
    "programming_languages": "programming languages",
    "projects": "projects",
    "goals": "goals",
    "preferences": "preferences",
    "hobbies": "hobbies",
    "tools": "tools and applications",
    "tasks": "tasks and daily items",
}


# ---------- Embeddings ----------

def get_embedding(text):
    """Generate normalized 768-dim float32 embedding vector using nomic-embed-text."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    vector = np.array(response["embeddings"][0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vector)
    return vector


# ---------- Knowledge Base RAG ----------

def load_knowledge_store():
    if not os.path.exists(KNOWLEDGE_INDEX_FILE) or not os.path.exists(KNOWLEDGE_CHUNKS_FILE):
        return None, []
    try:
        loaded_index = faiss.read_index(KNOWLEDGE_INDEX_FILE)
        with open(KNOWLEDGE_CHUNKS_FILE, "r", encoding="utf-8") as file:
            loaded_chunks = json.load(file)
        return loaded_index, loaded_chunks
    except Exception as error:
        print(f"Warning: could not load knowledge store: {error}")
        return None, []


knowledge_index, knowledge_chunks = load_knowledge_store()


def search_knowledge(query):
    if knowledge_index is None or not knowledge_chunks:
        return "Knowledge search is unavailable because the vector store is missing or empty."

    query_vec = get_embedding(query)
    scores, indices = knowledge_index.search(query_vec, min(TOP_K_KNOWLEDGE, len(knowledge_chunks)))

    results = []
    for index_id in indices[0]:
        if index_id == -1 or index_id >= len(knowledge_chunks):
            continue
        chunk = knowledge_chunks[index_id]
        results.append(f"Source: {chunk.get('source', 'Unknown')}\n{chunk.get('text', '')}")

    return "\n\n".join(results) if results else "No relevant information found."


# ---------- Declarative Fact Template & Extraction ----------

def fact_to_statement(category, item):
    templates = {
        "name": f"The user's name is {item}.",
        "age": f"The user is {item} years old.",
        "education": f"The user's education background is {item}.",
        "location": f"The user lives or is located in {item}.",
        "learning": f"The user is learning or studying {item}.",
        "skills": f"The user has skill and competence in {item}.",
        "programming_languages": f"The user codes in the programming language {item}.",
        "projects": f"The user is developing a project named {item}.",
        "goals": f"The user's objective or goal is {item}.",
        "preferences": f"The user prefers {item}.",
        "hobbies": f"The user's hobby or interest is {item}.",
        "tools": f"The user uses software tool or application {item}.",
        "tasks": f"The user has a current task or planned activity: {item}.",
    }
    return templates.get(category, f"The user's {category.replace('_', ' ')} is {item}.")


# ---------- Memory Lifecycle Store (Records + FAISS Index) ----------

memory_records = []
active_fact_ids = []
memory_index = None


def load_memory_store():
    global memory_records
    os.makedirs(MEMORY_STORE_DIR, exist_ok=True)
    if os.path.exists(MEMORY_RECORDS_FILE):
        try:
            with open(MEMORY_RECORDS_FILE, "r", encoding="utf-8") as file:
                memory_records = json.load(file)
        except Exception as error:
            print(f"Warning: could not load memory records: {error}")
            memory_records = []
    else:
        memory_records = []

    if not memory_records:
        migrated_records = []
        now_str = datetime.now(timezone.utc).isoformat()

        if os.path.exists(LEGACY_MEMORY_FILE):
            try:
                with open(LEGACY_MEMORY_FILE, "r", encoding="utf-8") as file:
                    legacy_data = json.load(file)
                if isinstance(legacy_data, dict):
                    for cat, val in legacy_data.items():
                        items = val if isinstance(val, list) else [val]
                        for item in items:
                            if item:
                                statement = fact_to_statement(cat, item)
                                migrated_records.append({
                                    "id": len(migrated_records),
                                    "category": cat,
                                    "item": item,
                                    "statement": statement,
                                    "created_at": now_str,
                                    "ttl_days": None,
                                    "is_active": True,
                                    "superseded_by": None,
                                    "provenance": "Migrated from legacy memory.json",
                                })
            except Exception as error:
                print(f"Warning: could not migrate from memory.json: {error}")

        if os.path.exists(V10_FACTS_FILE):
            try:
                with open(V10_FACTS_FILE, "r", encoding="utf-8") as file:
                    v10_data = json.load(file)
                if isinstance(v10_data, list):
                    existing_statements = {r["statement"].casefold() for r in migrated_records}
                    for f in v10_data:
                        stmt = f.get("statement", "")
                        if stmt and stmt.casefold() not in existing_statements:
                            migrated_records.append({
                                "id": len(migrated_records),
                                "category": f.get("category", "general"),
                                "item": f.get("item", ""),
                                "statement": stmt,
                                "created_at": now_str,
                                "ttl_days": None,
                                "is_active": True,
                                "superseded_by": None,
                                "provenance": "Migrated from v10 memory_facts.json",
                            })
                            existing_statements.add(stmt.casefold())
            except Exception as error:
                print(f"Warning: could not migrate from memory_facts.json: {error}")

        if migrated_records:
            memory_records = migrated_records
            save_memory_store()


def rebuild_active_index():
    global memory_index, active_fact_ids
    active_fact_ids = []
    active_statements = []

    for record in memory_records:
        if record.get("is_active", True):
            active_fact_ids.append(record["id"])
            active_statements.append(record["statement"])

    # If active index file exists on disk and matches count, load it directly
    if os.path.exists(MEMORY_INDEX_FILE):
        try:
            loaded_index = faiss.read_index(MEMORY_INDEX_FILE)
            if loaded_index.ntotal == len(active_statements):
                memory_index = loaded_index
                return
        except Exception:
            pass

    memory_index = faiss.IndexFlatIP(MEMORY_EMBED_DIM)
    if active_statements:
        try:
            vectors = [get_embedding(s) for s in active_statements]
            stacked = np.vstack(vectors).astype("float32")
            memory_index.add(stacked)
            faiss.write_index(memory_index, MEMORY_INDEX_FILE)
        except Exception as error:
            print(f"Warning: Could not connect to Ollama to embed memories: {error}")


def save_memory_store():
    temp_file = MEMORY_RECORDS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(memory_records, file, indent=2, ensure_ascii=False)
    os.replace(temp_file, MEMORY_RECORDS_FILE)
    rebuild_active_index()


def sweep_expired_memories():
    now = datetime.now(timezone.utc)
    expired_count = 0
    for record in memory_records:
        if not record.get("is_active", True):
            continue
        ttl_days = record.get("ttl_days")
        if ttl_days is not None:
            created_at = datetime.fromisoformat(record["created_at"])
            if now > created_at + timedelta(days=ttl_days):
                record["is_active"] = False
                record["superseded_by"] = "EXPIRED"
                expired_count += 1
    if expired_count > 0:
        save_memory_store()


load_memory_store()
sweep_expired_memories()
rebuild_active_index()


# ---------- Memory Extraction & Conflict Resolution ----------

def extract_memory_with_lifecycle(user_input):
    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": '''You are a careful memory-extraction system.
Extract user facts and classify if they are permanent or temporary.
Possible fields:
- name: string
- age: number
- education: string or list of strings
- location: string (city/state)
- learning: list of strings
- skills: list of strings
- programming_languages: list of strings
- projects: list of strings
- goals: list of strings
- preferences: list of strings
- hobbies: list of strings
- tools: list of strings
- tasks: list of strings (daily or weekly items)

Rules:
- Lists must ALWAYS be JSON arrays.
- Do not invent facts. Return ONLY valid JSON, with no markdown fences.
- Add an optional field "ephemeral": boolean for short-term daily items.
- If there is no fact, return {}.''',
                },
                {"role": "user", "content": user_input},
            ],
        )

        text = response.get("message", {}).get("content", "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        data = json.loads(text)

        if not isinstance(data, dict):
            return {}, False

        is_ephemeral = bool(data.pop("ephemeral", False))

        common_languages = {
            "python", "c++", "c", "java", "javascript", "typescript",
            "c#", "rust", "go", "ruby", "php", "swift", "kotlin", "sql", "html", "css"
        }
        if "projects" in data and isinstance(data["projects"], list):
            valid_projects = []
            for p in data["projects"]:
                if isinstance(p, str) and p.strip().lower() in common_languages:
                    data.setdefault("programming_languages", [])
                    if isinstance(data["programming_languages"], list) and p not in data["programming_languages"]:
                        data["programming_languages"].append(p)
                else:
                    valid_projects.append(p)
            data["projects"] = valid_projects

        allowed_fields = set(MEMORY_LABELS)
        cleaned = {
            key: value
            for key, value in data.items()
            if key in allowed_fields and value not in (None, "", [], {})
        }
        return cleaned, is_ephemeral

    except Exception:
        return {}, False


def judge_conflict(existing_statement, new_statement, user_input):
    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI Memory Conflict Judge.
Decide if New Fact REPLACEs or should KEEP_BOTH with Existing Fact.
Respond ONLY with valid JSON: {"action": "REPLACE" or "KEEP_BOTH", "reason": "brief reason"}""",
                },
                {
                    "role": "user",
                    "content": f"User: \"{user_input}\"\nExisting: \"{existing_statement}\"\nNew: \"{new_statement}\"",
                },
            ],
        )
        text = response.get("message", {}).get("content", "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        decision = json.loads(text)
        action = decision.get("action", "KEEP_BOTH").upper()
        return action if action in {"REPLACE", "KEEP_BOTH"} else "KEEP_BOTH"
    except Exception:
        return "KEEP_BOTH"


def update_memory_with_lifecycle(user_input):
    extracted, is_ephemeral = extract_memory_with_lifecycle(user_input)
    if not extracted:
        return {}

    newly_added = {}
    records_to_add = []
    conflict_actions = []

    for key, value in extracted.items():
        items = value if isinstance(value, list) else [value]
        added_for_key = []

        for item in items:
            if not str(item).strip():
                continue
            new_statement = fact_to_statement(key, item)

            already_active = any(
                r.get("is_active", True) and r["statement"].casefold() == new_statement.casefold()
                for r in memory_records
            )
            if already_active:
                continue

            conflicting_record = None
            if memory_index is not None and memory_index.ntotal > 0:
                vec = get_embedding(new_statement)
                scores, indices = memory_index.search(vec, min(3, memory_index.ntotal))

                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or idx >= len(active_fact_ids):
                        continue
                    if score >= CONFLICT_CHECK_THRESHOLD:
                        candidate_id = active_fact_ids[idx]
                        candidate_record = next((r for r in memory_records if r["id"] == candidate_id), None)
                        if candidate_record and candidate_record.get("is_active", True):
                            if candidate_record["category"] == key or key not in LIST_FIELDS:
                                action = judge_conflict(candidate_record["statement"], new_statement, user_input)
                                if action == "REPLACE":
                                    conflicting_record = candidate_record
                                    break

            new_record_id = len(memory_records) + len(records_to_add)
            ttl = 7 if (is_ephemeral or key == "tasks") else None

            if conflicting_record is not None:
                conflicting_record["is_active"] = False
                conflicting_record["superseded_by"] = new_record_id
                conflict_actions.append(
                    f"Fact ID {conflicting_record['id']} ('{conflicting_record['statement']}') "
                    f"superseded by ID {new_record_id} ('{new_statement}')"
                )

            new_record = {
                "id": new_record_id,
                "category": key,
                "item": item,
                "statement": new_statement,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ttl_days": ttl,
                "is_active": True,
                "superseded_by": None,
                "provenance": user_input,
            }

            records_to_add.append(new_record)
            added_for_key.append(item)

        if added_for_key:
            newly_added[key] = added_for_key

    if records_to_add or conflict_actions:
        memory_records.extend(records_to_add)
        save_memory_store()
        if conflict_actions:
            print("\n[Memory Lifecycle: Conflict Resolved]")
            for ca in conflict_actions:
                print(f"  • {ca}")

    return newly_added


def retrieve_relevant_memories(query, top_k=TOP_K_MEMORY, threshold=MEMORY_SIMILARITY_THRESHOLD):
    if memory_index is None or memory_index.ntotal == 0:
        return []

    query_vec = get_embedding(query)
    search_k = min(top_k, memory_index.ntotal)
    scores, indices = memory_index.search(query_vec, search_k)

    relevant = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(active_fact_ids):
            continue
        if score >= threshold:
            fact_id = active_fact_ids[idx]
            fact = next((r for r in memory_records if r["id"] == fact_id and r.get("is_active", True)), None)
            if fact:
                relevant.append({
                    "id": fact["id"],
                    "statement": fact["statement"],
                    "category": fact["category"],
                    "score": float(score),
                    "provenance": fact.get("provenance", ""),
                })

    return relevant


# ---------- Tools & Calculator ----------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval_math(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent is too large.")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval_math(node.operand))
    raise ValueError("Only basic arithmetic expressions are allowed.")


def calculator(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval_math(tree))
    except Exception as error:
        return f"ERROR: {error}"


def weather(city):
    return f"Weather lookup is not connected to a live weather service for {city}."


def forget_memory(query):
    if memory_index is None or memory_index.ntotal == 0:
        return "Memory is currently empty. Nothing to forget."

    vec = get_embedding(query)
    scores, indices = memory_index.search(vec, min(5, memory_index.ntotal))

    forgotten = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(active_fact_ids):
            continue
        if score >= 0.50:
            rec_id = active_fact_ids[idx]
            rec = next((r for r in memory_records if r["id"] == rec_id and r.get("is_active", True)), None)
            if rec:
                rec["is_active"] = False
                rec["superseded_by"] = "USER_FORGOTTEN"
                forgotten.append(f"ID {rec['id']}: '{rec['statement']}'")

    if forgotten:
        save_memory_store()
        return "Successfully forgotten the following facts:\n- " + "\n- ".join(forgotten)
    return f"Could not find any active memories matching '{query}' to forget."


tools = {
    "calculator": calculator,
    "weather": weather,
    "search_knowledge": search_knowledge,
    "forget_memory": forget_memory,
}

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Reports whether live weather lookup is available for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Searches the local knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Surgically deletes or forgets user facts from long-term memory upon request.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def execute_tool(name, arguments):
    if name not in tools:
        return f"ERROR: Tool '{name}' does not exist."
    try:
        return str(tools[name](**arguments))
    except Exception as error:
        return f"ERROR: {error}"


def verify_tool_result(name, result):
    if result.startswith("ERROR:"):
        return "RETRY"
    if name in {"weather", "calculator", "forget_memory"}:
        return "VALID"
    if name == "search_knowledge":
        if "No relevant information found" in result or "unavailable" in result:
            return "RETRY"
    return "VALID"


# ---------- PILLAR 5: DAG PLANNER & STATE MACHINE ENGINE ----------

def is_complex_goal(user_input):
    """
    Classify whether a user input requires multi-step DAG planning vs single-turn ReAct.
    """
    text = user_input.lower()
    markers = [
        " and then ", " after that ", "step by step", "first ", "research and",
        "calculate and", "search and calculate", "plan:", "mission:", "workflow"
    ]
    if any(m in text for m in markers) or len(text.split()) > 18:
        return True
    return False


def generate_dag_plan(goal_text, memory_context=""):
    """
    Ask Qwen3 to decompose a complex goal into an executable Directed Acyclic Graph (DAG).
    Each step specifies its task description, tool to execute, arguments, and dependencies.
    """
    current_year = datetime.now().year
    system_prompt = f"""You are an Autonomous AI Planning Architect.
Current Year: {current_year}
Decompose the user's high-level goal into an optimal, non-redundant Directed Acyclic Graph (DAG) of executable steps.
{memory_context}

Available tools:
- "search_knowledge": arguments {{"query": string}} - searches internal knowledge base
- "calculator": arguments {{"expression": string}} - solves math expressions (must contain ONLY numbers and arithmetic operators, e.g. "21 + 2" or "2028 - 2005")
- "weather": arguments {{"city": string}} - weather lookup
- "none": arguments {{}} - used for pure synthesis, deduction, or formatting from previous step results

Rules:
1. Grounding in Memory: Use known facts from the user's memory when available rather than re-asking the user.
2. If mathematical calculations are needed, substitute known numbers from memory directly into the calculator "expression".
3. Break down into 2 to 4 clear, logical steps.
4. For each step, define "depends_on" as a JSON array of step IDs that must complete before this step runs.
5. The final step must have tool: "none" to synthesize the entire goal and present the final answer.
6. Return ONLY a valid JSON object matching this schema:
{{
  "steps": [
    {{
      "id": 1,
      "task": "Description of what to execute",
      "tool": "search_knowledge" | "calculator" | "weather" | "none",
      "arguments": {{ ... }},
      "depends_on": []
    }}
  ]
}}"""

    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Goal: {goal_text}"},
            ],
        )
        raw_text = response.get("message", {}).get("content", "").strip()
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
        data = json.loads(raw_text)
        steps = data.get("steps", [])

        # Validate DAG structure
        validated_steps = []
        for s in steps:
            validated_steps.append({
                "id": int(s.get("id", len(validated_steps) + 1)),
                "task": str(s.get("task", "")),
                "tool": str(s.get("tool", "none")),
                "arguments": s.get("arguments", {}) if isinstance(s.get("arguments"), dict) else {},
                "depends_on": [int(d) for d in s.get("depends_on", []) if isinstance(d, (int, str)) and str(d).isdigit()],
                "status": "PENDING",
                "result": None,
                "error": None,
            })
        return validated_steps
    except Exception as error:
        print(f"Plan generation fallback: {error}")
        return [
            {
                "id": 1,
                "task": f"Direct execution of: {goal_text}",
                "tool": "none",
                "arguments": {},
                "depends_on": [],
                "status": "PENDING",
                "result": None,
                "error": None,
            }
        ]


def replan_dag(goal_text, failed_step, completed_steps, remaining_steps, memory_context=""):
    """
    Dynamic Replanner: If a step fails, Qwen3 inspects the failure and rewrites the remaining DAG.
    """
    print(f"\n[Dynamic Replanner Triggered] Step {failed_step['id']} failed with error: {failed_step['error']}")
    print("Synthesizing alternative recovery plan...")

    context = {
        "goal": goal_text,
        "failed_step": failed_step,
        "completed_steps": [{"id": s["id"], "task": s["task"], "result": str(s["result"])[:150]} for s in completed_steps],
        "remaining_steps": [{"id": s["id"], "task": s["task"]} for s in remaining_steps],
    }

    prompt = f"""You are an AI Replanning Engine.
A step in an active plan failed. Reroute and rewrite the remaining steps to achieve the goal despite this failure.
{memory_context}

Plan Context:
{json.dumps(context, indent=2)}

Return a new JSON object for the replacement remaining steps:
{{
  "steps": [
    {{
      "id": <start from {failed_step['id']}>,
      "task": "Revised task description",
      "tool": "search_knowledge" | "calculator" | "weather" | "none",
      "arguments": {{ ... }},
      "depends_on": [<completed step IDs>]
    }}
  ]
}}"""

    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {"role": "system", "content": "You are an autonomous replanner. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_text = response.get("message", {}).get("content", "").strip()
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
        data = json.loads(raw_text)
        new_steps = data.get("steps", [])

        validated = []
        for s in new_steps:
            validated.append({
                "id": int(s.get("id", len(completed_steps) + len(validated) + 1)),
                "task": str(s.get("task", "")),
                "tool": str(s.get("tool", "none")),
                "arguments": s.get("arguments", {}) if isinstance(s.get("arguments"), dict) else {},
                "depends_on": [int(d) for d in s.get("depends_on", []) if isinstance(d, (int, str)) and str(d).isdigit()],
                "status": "PENDING",
                "result": None,
                "error": None,
            })
        return validated
    except Exception as error:
        print(f"Replanning error: {error}. Falling back to error synthesis step.")
        return [
            {
                "id": failed_step["id"],
                "task": f"Report limitation: could not complete {failed_step['task']} due to {failed_step['error']}",
                "tool": "none",
                "arguments": {},
                "depends_on": [s["id"] for s in completed_steps],
                "status": "PENDING",
                "result": None,
                "error": None,
            }
        ]


def display_checklist(steps):
    """Render a live, Devin-style checklist in terminal."""
    status_icons = {
        "COMPLETED": "[OK]",
        "RUNNING": "[*]",
        "PENDING": "[ ]",
        "FAILED": "[X]",
        "SKIPPED": "[-]",
    }
    print("\n" + "=" * 65)
    print("[PLAN EXECUTION CHECKLIST]")
    for s in steps:
        icon = status_icons.get(s["status"], "[ ]")
        deps = f" (depends on: {s['depends_on']})" if s['depends_on'] else ""
        print(f"  {icon} Step {s['id']}: {s['task']}{deps} [{s['status']}]")
    print("=" * 65)


def execute_dag_plan(goal_text, steps, memory_context=""):
    """
    Deterministic State Machine Executor with Invariant Checking & Replanning.
    """
    replan_count = 0
    all_steps = list(steps)

    while True:
        display_checklist(all_steps)

        # Find next ready step (status PENDING and all dependencies COMPLETED)
        completed_ids = {s["id"] for s in all_steps if s["status"] == "COMPLETED"}
        ready_step = None
        for s in all_steps:
            if s["status"] == "PENDING":
                if all(dep in completed_ids for dep in s["depends_on"]):
                    ready_step = s
                    break

        if ready_step is None:
            # Check if any pending steps remain blocked
            pending_steps = [s for s in all_steps if s["status"] == "PENDING"]
            if pending_steps:
                print(f"\n[Deadlock Detected] Unmet dependencies for remaining steps: {[s['id'] for s in pending_steps]}")
            break

        # Transition state: PENDING -> RUNNING
        ready_step["status"] = "RUNNING"
        display_checklist(all_steps)

        print(f"\n--> Executing Step {ready_step['id']}: {ready_step['task']}")
        step_tool = ready_step["tool"]
        step_args = ready_step["arguments"]

        # Inject context from prior steps if arguments reference placeholder
        context_summary = "\n".join(
            f"Step {s['id']} ({s['task']}) result: {s['result']}"
            for s in all_steps if s['status'] == "COMPLETED" and s['result']
        )

        step_success = False
        step_result = None
        step_error = None

        if step_tool in tools:
            # Tool-based step
            result = execute_tool(step_tool, step_args)
            print(f"Tool [{step_tool}] Result: {result[:200]}...")
            decision = verify_tool_result(step_tool, result)

            if decision == "RETRY" or result.startswith("ERROR:"):
                step_error = result
                step_success = False
            else:
                step_result = result
                step_success = True
        else:
            # Synthesis / reasoning step using Qwen3
            synthesis_prompt = f"""You are synthesizing the results of an autonomous plan.
Original Goal: {goal_text}
Step Objective: {ready_step['task']}
{memory_context}

Prior Step Outputs:
{context_summary}

Synthesize a comprehensive, clear, and direct response that completes this step and satisfies the user's goal."""
            try:
                response = ollama.chat(
                    model="qwen3",
                    messages=[
                        {"role": "system", "content": "You are a precise task synthesizer."},
                        {"role": "user", "content": synthesis_prompt},
                    ],
                )
                step_result = response.get("message", {}).get("content", "").strip()
                step_success = True
            except Exception as error:
                step_error = str(error)
                step_success = False

        if step_success:
            ready_step["status"] = "COMPLETED"
            ready_step["result"] = step_result
        else:
            ready_step["status"] = "FAILED"
            ready_step["error"] = step_error
            print(f"\n[X] Step {ready_step['id']} failed: {step_error}")

            if replan_count < MAX_REPLAN_ATTEMPTS:
                replan_count += 1
                completed = [s for s in all_steps if s["status"] == "COMPLETED"]
                remaining = [s for s in all_steps if s["status"] == "PENDING"]
                new_steps = replan_dag(goal_text, ready_step, completed, remaining, memory_context=memory_context)

                # Reconstruct DAG with completed steps + new patched steps
                all_steps = completed + [ready_step] + new_steps
                continue
            else:
                print("\n[Exceeded Replan Limit] Terminating plan execution.")
                break

    # Final summary output
    display_checklist(all_steps)
    completed_steps = [s for s in all_steps if s["status"] == "COMPLETED"]
    last_step = completed_steps[-1] if completed_steps else None

    if last_step and last_step.get("result"):
        final_answer = last_step["result"]
    else:
        final_answer = f"Plan partially completed ({len(completed_steps)}/{len(all_steps)} steps succeeded)."

    return final_answer


# ---------- Main Agent Dispatcher ----------

conversation_history = []


def run_agent(user_input):
    sweep_expired_memories()
    update_memory_with_lifecycle(user_input)

    # 1. Retrieve relevant long-term memories first
    relevant_memories = retrieve_relevant_memories(user_input)
    memory_context = ""
    if relevant_memories:
        print(f"\n[Semantic Memory] Retrieved {len(relevant_memories)} active facts (Threshold >= {MEMORY_SIMILARITY_THRESHOLD}):")
        for mem in relevant_memories:
            print(f"  * ({mem['score']:.3f}) [ID {mem['id']}] {mem['statement']}")
        facts_text = "\n".join(f"- {mem['statement']}" for mem in relevant_memories)
        memory_context = f"\n\nKnown User Facts retrieved from long-term memory:\n{facts_text}"
    else:
        print("\n[Semantic Memory] No active facts were sufficiently relevant. Zero memory injected.")

    # 2. Check if user query is a complex multi-step mission requiring DAG planning
    if is_complex_goal(user_input):
        print("\n[Planner Activated] Decomposing complex mission into dependency DAG...")
        steps = generate_dag_plan(user_input, memory_context=memory_context)
        answer = execute_dag_plan(user_input, steps, memory_context=memory_context)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})
        return answer

    # 3. Otherwise, standard conversational & semantic memory loop
    system_prompt = f"""You are an autonomous AI agent.
Use the retrieved user memories when relevant to personalize your response, but do not invent facts.
Use search_knowledge for questions about the local knowledge base.
Use calculator for mathematical calculations.
Use forget_memory if the user explicitly asks you to forget a specific fact.
For weather, clearly state that live weather is unavailable unless a live service is connected.{memory_context}"""

    conversation_history.append({"role": "user", "content": user_input})
    recent_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    messages = [
        {"role": "system", "content": system_prompt},
        *recent_history,
    ]

    retries = 0
    for _ in range(MAX_ITERATIONS):
        try:
            response = ollama.chat(model="qwen3", messages=messages, tools=tool_definitions)
        except Exception as error:
            if conversation_history:
                conversation_history.pop()
            return f"Could not contact Ollama. Details: {error}"

        message = response.get("message", {})
        messages.append(message)
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            final_content = message.get("content", "")
            conversation_history.append({"role": "assistant", "content": final_content})
            return final_content

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            if not isinstance(arguments, dict):
                return "Tool call failed: arguments were not a JSON object."

            print(f"\nTool: {name}")
            print(f"Arguments: {arguments}")
            result = execute_tool(name, arguments)
            print(f"Result: {result}")

            decision = verify_tool_result(name, result)
            print(f"Verification: {decision}")

            if decision == "RETRY":
                retries += 1
                if retries > MAX_RETRIES:
                    fail_msg = "Tool failed after several attempts."
                    conversation_history.append({"role": "assistant", "content": fail_msg})
                    return fail_msg
                messages.append({
                    "role": "user",
                    "content": f"The tool result was not useful: {result}. Try a different approach or explain the limitation.",
                })
                continue

            messages.append({"role": "tool", "tool_name": name, "content": result})

    fallback_msg = "Maximum iterations reached."
    conversation_history.append({"role": "assistant", "content": fallback_msg})
    return fallback_msg


# ---------- Main ----------

if __name__ == "__main__":
    print("=" * 75)
    print("AI Agent V12 (Reliable Planning, Preconditions & State Recovery) started.")
    print("Commands:")
    print("  /plan <goal>         - Explicitly trigger DAG decomposition & execution")
    print("  /memory              - List active memories & TTLs")
    print("  /memory_all          - Audit full memory lifecycle ledger")
    print("  /clear_memory        - Wipe memory store")
    print("  /clear_history       - Wipe conversation history")
    print("  exit / quit          - Exit agent")
    print("=" * 75)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting agent.")
            break

        if not user_input:
            continue
        if user_input.casefold() in {"exit", "quit"}:
            break
        if user_input == "/memory":
            active = [r for r in memory_records if r.get("is_active", True)]
            print(f"\n--- ACTIVE MEMORY ({len(active)} facts) ---")
            for r in active:
                ttl = f"TTL: {r['ttl_days']}d" if r.get('ttl_days') else "Permanent"
                print(f"[{r['id']}] ({r['category']}) [{ttl}]: {r['statement']}")
            continue
        if user_input == "/memory_all":
            print(f"\n--- COMPLETE MEMORY AUDIT LEDGER ({len(memory_records)} records) ---")
            for r in memory_records:
                status = "ACTIVE" if r.get("is_active", True) else f"INACTIVE ({r.get('superseded_by')})"
                print(f"[{r['id']}] [{status}]: {r['statement']}")
            continue
        if user_input.startswith("/plan"):
            goal = user_input.replace("/plan", "", 1).strip()
            if goal:
                print(f"\n[Explicit Plan Mode] Goal: {goal}")
                rel_mems = retrieve_relevant_memories(goal)
                mem_ctx = ""
                if rel_mems:
                    f_text = "\n".join(f"- {m['statement']}" for m in rel_mems)
                    mem_ctx = f"\n\nKnown User Facts retrieved from long-term memory:\n{f_text}"
                dag_steps = generate_dag_plan(goal, memory_context=mem_ctx)
                ans = execute_dag_plan(goal, dag_steps, memory_context=mem_ctx)
                print("\nAgent:", ans)
            else:
                print("Usage: /plan <your multi-step goal>")
            continue
        if user_input == "/clear_memory":
            memory_records = []
            save_memory_store()
            print("Memory cleared.")
            continue
        if user_input == "/clear_history":
            conversation_history.clear()
            print("Conversation history cleared.")
            continue

        answer = run_agent(user_input)
        print("\nAgent:", answer)

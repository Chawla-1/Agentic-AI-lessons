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


# ---------- Declarative Fact Template & Declarations ----------

def fact_to_statement(category, item):
    """Convert an atomic fact item into a clear declarative sentence."""
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

memory_records = []         # Complete ledger of all memory facts (active, superseded, expired, forgotten)
active_fact_ids = []        # Maps FAISS index row i -> memory_records id
memory_index = None         # FAISS index indexing ONLY active facts


def load_memory_store():
    """Load the lifecycle records from disk, or auto-migrate legacy memories if store is empty."""
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

    # Auto-migration: if memory_records is empty, import legacy memory files seamlessly
    if not memory_records:
        migrated_records = []
        now_str = datetime.now(timezone.utc).isoformat()

        # 1. Check legacy memory.json (from v09)
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

        # 2. Check legacy memory_facts.json (from v10)
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
            print(f"\n[System Migration] Auto-migrated {len(memory_records)} legacy facts into active V11 lifecycle records.")
            save_memory_store()


def rebuild_active_index():
    """
    Rebuild the FAISS index so it contains ONLY currently active, non-expired facts.
    Maintains active_fact_ids mapping.
    """
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
    """Atomically persist memory records and refresh the FAISS index."""
    temp_file = MEMORY_RECORDS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(memory_records, file, indent=2, ensure_ascii=False)
    os.replace(temp_file, MEMORY_RECORDS_FILE)
    rebuild_active_index()


def sweep_expired_memories():
    """Check TTL on all active facts. Deactivate any whose expiration timestamp has passed."""
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
                print(f"[Memory Lifecycle: Expired] Fact ID {record['id']} expired: '{record['statement']}'")

    if expired_count > 0:
        save_memory_store()


load_memory_store()
sweep_expired_memories()
rebuild_active_index()


# ---------- Memory Extraction & Ephemeral Classification ----------


def extract_memory_with_lifecycle(user_input):
    """
    Ask Qwen3 to extract structured facts and assess whether any fact is ephemeral (has TTL).
    """
    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": '''You are a careful memory-extraction system.
Extract user facts and classify if they are permanent or temporary.

Possible fields:
- name: string (permanent)
- age: number (permanent)
- education: string or list of strings (permanent)
- location: string (user current living city/state)
- learning: list of strings (permanent/long-term)
- skills: list of strings (permanent)
- programming_languages: list of strings (permanent)
- projects: list of strings (long-term)
- goals: list of strings (long-term)
- preferences: list of strings (long-term)
- hobbies: list of strings (long-term)
- tools: list of strings (software, IDEs, databases)
- tasks: list of strings (daily or weekly ephemeral items, e.g. "two tasks for today: coding and playing")

Rules:
- For learning, skills, programming_languages, projects, goals, preferences, hobbies, tools, and tasks, ALWAYS return a JSON list.
- If the user mentions working on a project IN a programming language (e.g. "project in Java"), assign that language to 'programming_languages', NOT 'projects'.
- Do not invent facts. Return ONLY valid JSON, with no markdown fences.
- Add an optional field "ephemeral": boolean. Set "ephemeral": true ONLY if the fact is a temporary day/week item (e.g. daily tasks, illness, immediate trip). Otherwise false.
- If there is no user fact, return {}.

Examples:
User: My name is Aaryan. I live in Delhi.
Output: {"name": "Aaryan", "location": "Delhi", "ephemeral": false}
User: I am 22 years old now.
Output: {"age": 22, "ephemeral": false}
User: I have two tasks for today: coding and playing.
Output: {"tasks": ["coding", "playing"], "ephemeral": true}
User: What is machine learning?
Output: {}''',
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

        # Algorithmic safety guard: prevent programming languages from lingering in projects
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


# ---------- Conflict Resolution Engine ----------

def judge_conflict(existing_statement, new_statement, user_input):
    """
    Ask Qwen3 to act as a judge to decide if a new fact CONTRADICTS / REPLACES an old fact,
    or if both should be KEPT (addition / distinct items).
    """
    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI Memory Conflict Judge.
Compare an Existing Fact with a New Fact given the user's latest message.

Decide one of two actions:
1. "REPLACE": The new fact updates, corrects, or directly contradicts the existing fact.
   Examples:
   - Existing: "User lives in Delhi" vs New: "User lives in Bangalore" -> REPLACE
   - Existing: "User is 21 years old" vs New: "User is 22 years old" -> REPLACE
   - Existing: "User is working at Google" vs New: "User works at a startup" -> REPLACE
   - Existing: "User's goal is to be a frontend dev" vs New: "User's goal is AI research" -> REPLACE

2. "KEEP_BOTH": The new fact is an addition, an extra skill/tool/hobby, or a separate distinct item.
   Examples:
   - Existing: "User knows Python" vs New: "User knows C++" -> KEEP_BOTH
   - Existing: "User uses VS Code" vs New: "User uses Android Studio" -> KEEP_BOTH
   - Existing: "User hobby is chess" vs New: "User hobby is cricket" -> KEEP_BOTH

Respond ONLY with valid JSON: {"action": "REPLACE" or "KEEP_BOTH", "reason": "brief reason"}""",
                },
                {
                    "role": "user",
                    "content": f"User utterance: \"{user_input}\"\nExisting Fact: \"{existing_statement}\"\nNew Fact: \"{new_statement}\"",
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
    """
    Extract facts, check for conflicts/superseding, assign TTLs, and update memory store.
    """
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

            # Check if this exact statement already exists as an active fact
            already_active = any(
                r.get("is_active", True) and r["statement"].casefold() == new_statement.casefold()
                for r in memory_records
            )
            if already_active:
                continue

            # Candidate conflict detection using vector similarity against active facts
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
                            # Same category or high similarity -> Run judge
                            if candidate_record["category"] == key or key not in LIST_FIELDS:
                                action = judge_conflict(candidate_record["statement"], new_statement, user_input)
                                if action == "REPLACE":
                                    conflicting_record = candidate_record
                                    break

            new_record_id = len(memory_records) + len(records_to_add)
            ttl = 7 if (is_ephemeral or key == "tasks") else None

            if conflicting_record is not None:
                # Resolve conflict by marking old fact as superseded
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

        if records_to_add:
            print("\n[Memory Lifecycle: New Facts Stored]")
            for r in records_to_add:
                ttl_str = f"TTL: {r['ttl_days']} days" if r['ttl_days'] else "Permanent"
                print(f"  • [ID {r['id']}] ({r['category']}) '{r['statement']}' ({ttl_str})")

    return newly_added


# ---------- Surgical Forgetting & Provenance ----------

def forget_memory(query):
    """
    Surgically deactivate memories semantically matching the query.
    Implements Right-to-be-Forgotten.
    """
    if memory_index is None or memory_index.ntotal == 0:
        return "Memory is currently empty. Nothing to forget."

    vec = get_embedding(query)
    scores, indices = memory_index.search(vec, min(5, memory_index.ntotal))

    forgotten = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(active_fact_ids):
            continue
        if score >= 0.50:  # Threshold for forget matching
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


def get_provenance(query):
    """Show the original context, timestamp, and source utterance for a memory."""
    if not memory_records:
        return "No memories recorded."

    vec = get_embedding(query)
    # Search all records
    results = []
    for r in memory_records:
        r_vec = get_embedding(r["statement"])
        sim = float(np.dot(vec, r_vec.T)[0][0])
        if sim >= 0.45:
            results.append((sim, r))

    results.sort(key=lambda x: x[0], reverse=True)
    if not results:
        return f"No memories found relating to '{query}'."

    output = [f"Provenance audit for '{query}':\n"]
    for sim, r in results[:3]:
        status = "ACTIVE" if r.get("is_active", True) else f"INACTIVE ({r.get('superseded_by')})"
        output.append(
            f"• [ID {r['id']}] '{r['statement']}'\n"
            f"  Status: {status} | Similarity: {sim:.3f}\n"
            f"  Created: {r['created_at']}\n"
            f"  Original User Quote: \"{r.get('provenance', 'Unknown')}\"\n"
        )
    return "\n".join(output)


# ---------- Retrieval ----------

def retrieve_relevant_memories(query, top_k=TOP_K_MEMORY, threshold=MEMORY_SIMILARITY_THRESHOLD):
    """
    Search FAISS for active memories matching query above threshold.
    """
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


def show_active_memory():
    active = [r for r in memory_records if r.get("is_active", True)]
    if not active:
        print("Semantic memory is empty (no active facts).")
        return
    print(f"\n--- ACTIVE LONG-TERM MEMORY ({len(active)} active facts) ---")
    for r in active:
        ttl = f"TTL: {r['ttl_days']}d" if r.get('ttl_days') else "Permanent"
        print(f"[{r['id']}] ({r['category']}) [{ttl}]: {r['statement']}")


def show_all_memory():
    if not memory_records:
        print("Memory ledger is completely empty.")
        return
    print(f"\n--- COMPLETE MEMORY AUDIT LEDGER ({len(memory_records)} total records) ---")
    for r in memory_records:
        status = "ACTIVE" if r.get("is_active", True) else f"INACTIVE (by: {r.get('superseded_by')})"
        print(f"[{r['id']}] [{status}] ({r['category']}): {r['statement']}")


def clear_memory():
    global memory_records
    memory_records = []
    save_memory_store()
    print("Semantic memory and lifecycle ledger completely wiped.")




# ---------- Calculator ----------

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


# ---------- Weather ----------

def weather(city):
    return f"Weather lookup is not connected to a live weather service for {city}."


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
                "properties": {"query": {"type": "string", "description": "Topic or fact to forget"}},
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




# ---------- Agent Loop ----------

conversation_history = []


def run_agent(user_input):
    # 0. Sweep any expired memories
    sweep_expired_memories()

    # 1. Update memory with lifecycle & conflict checks
    newly_added = update_memory_with_lifecycle(user_input)

    # 4. Semantic retrieval of active facts
    relevant_memories = retrieve_relevant_memories(user_input)

    memory_context = ""
    if relevant_memories:
        print(f"\n[Semantic Memory] Retrieved {len(relevant_memories)} active facts (Threshold >= {MEMORY_SIMILARITY_THRESHOLD}):")
        for mem in relevant_memories:
            print(f"  • ({mem['score']:.3f}) [ID {mem['id']}] {mem['statement']}")
        facts_text = "\n".join(f"- {mem['statement']}" for mem in relevant_memories)
        memory_context = f"\n\nRelevant user facts retrieved from long-term memory:\n{facts_text}"
    else:
        print("\n[Semantic Memory] No active facts were sufficiently relevant. Zero memory injected.")

    new_facts_note = ""
    if newly_added:
        new_facts_note = (
            f"\n\nNOTICE: The user just shared new facts in this turn: {json.dumps(newly_added, ensure_ascii=False)}. "
            "Acknowledge them naturally and confirm what was just saved/updated."
        )

    system_prompt = f"""You are an AI agent.
Use the retrieved user memories when relevant to personalize your response, but do not invent facts.
Use search_knowledge for questions about the local knowledge base.
Use calculator for mathematical calculations.
Use forget_memory if the user explicitly asks you to forget or erase a specific fact or preference.
For weather, clearly state that live weather is unavailable unless a live service is connected.{memory_context}{new_facts_note}"""

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
    print("AI Agent V11 (Memory Lifecycle & Conflict Resolution) started.")
    print("Lifecycle Commands:")
    print("  /memory              - List currently ACTIVE memories & TTLs")
    print("  /memory_all          - Audit FULL ledger (including superseded & expired)")
    print("  /forget <topic>      - Surgically forget / deactivate a memory")
    print("  /provenance <topic>  - Inspect original quote, timestamp & source")
    print("  /clear_memory        - Wipe complete memory store")
    print("  /clear_history       - Wipe current session conversation history")
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
            show_active_memory()
            continue
        if user_input == "/memory_all":
            show_all_memory()
            continue
        if user_input.startswith("/forget"):
            topic = user_input.replace("/forget", "", 1).strip()
            if topic:
                res = forget_memory(topic)
                print(res)
            else:
                print("Usage: /forget <topic or fact>")
            continue
        if user_input.startswith("/provenance"):
            topic = user_input.replace("/provenance", "", 1).strip()
            if topic:
                print(get_provenance(topic))
            else:
                print("Usage: /provenance <topic or fact>")
            continue
        if user_input == "/clear_memory":
            clear_memory()
            continue
        if user_input == "/clear_history":
            conversation_history.clear()
            print("Session conversation history cleared.")
            continue

        answer = run_agent(user_input)
        print("\nAgent:", answer)


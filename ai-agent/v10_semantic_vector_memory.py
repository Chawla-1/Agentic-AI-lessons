import os
import json
import re
import ast
import operator
import ollama
import faiss
import numpy as np


MAX_ITERATIONS = 10
MAX_RETRIES = 2
MAX_HISTORY_TURNS = 10

# Knowledge base RAG paths
KNOWLEDGE_STORE_DIR = "vector_store"
KNOWLEDGE_INDEX_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "index.faiss")
KNOWLEDGE_CHUNKS_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "chunks.json")
TOP_K_KNOWLEDGE = 3

# Semantic Memory Store paths
MEMORY_STORE_DIR = "memory_store"
MEMORY_INDEX_FILE = os.path.join(MEMORY_STORE_DIR, "memory_index.faiss")
MEMORY_FACTS_FILE = os.path.join(MEMORY_STORE_DIR, "memory_facts.json")
MEMORY_EMBED_DIM = 768
TOP_K_MEMORY = 3
MEMORY_SIMILARITY_THRESHOLD = 0.52  # Cosine similarity cutoff for relevance

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
    "learning": "learning",
    "skills": "skills",
    "programming_languages": "programming languages",
    "projects": "projects",
    "goals": "goals",
    "preferences": "preferences",
    "hobbies": "hobbies",
    "tools": "tools and applications",
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


# ---------- Semantic Long-Term Memory Store ----------

def init_memory_store():
    """Load or initialize the atomic facts and FAISS index for user long-term memory."""
    os.makedirs(MEMORY_STORE_DIR, exist_ok=True)

    facts = []
    if os.path.exists(MEMORY_FACTS_FILE):
        try:
            with open(MEMORY_FACTS_FILE, "r", encoding="utf-8") as file:
                facts = json.load(file)
        except Exception as error:
            print(f"Warning: could not load memory_facts.json: {error}")
            facts = []

    index = None
    if os.path.exists(MEMORY_INDEX_FILE):
        try:
            index = faiss.read_index(MEMORY_INDEX_FILE)
        except Exception as error:
            print(f"Warning: could not load memory_index.faiss: {error}")
            index = None

    if index is None or index.ntotal != len(facts):
        # Rebuild index from facts if missing, corrupted, or desynced
        index = faiss.IndexFlatIP(MEMORY_EMBED_DIM)
        if facts:
            vectors = [get_embedding(item["statement"]) for item in facts]
            stacked = np.vstack(vectors).astype("float32")
            index.add(stacked)
            faiss.write_index(index, MEMORY_INDEX_FILE)

    return index, facts


memory_index, memory_facts = init_memory_store()


def save_memory_store():
    """Atomically persist memory facts and FAISS index."""
    temp_facts_file = MEMORY_FACTS_FILE + ".tmp"
    with open(temp_facts_file, "w", encoding="utf-8") as file:
        json.dump(memory_facts, file, indent=2, ensure_ascii=False)
    os.replace(temp_facts_file, MEMORY_FACTS_FILE)

    faiss.write_index(memory_index, MEMORY_INDEX_FILE)


def fact_to_statement(category, item):
    """Convert an atomic fact item into a clear, searchable declarative sentence."""
    templates = {
        "name": f"The user's name is {item}.",
        "age": f"The user is {item} years old.",
        "education": f"The user's education background is {item}.",
        "learning": f"The user is learning or studying {item}.",
        "skills": f"The user has skill and competence in {item}.",
        "programming_languages": f"The user codes in the programming language {item}.",
        "projects": f"The user is developing a project named {item}.",
        "goals": f"The user's objective or goal is {item}.",
        "preferences": f"The user prefers {item}.",
        "hobbies": f"The user's hobby or interest is {item}.",
        "tools": f"The user uses software tool or application {item}.",
    }
    return templates.get(category, f"The user's {category.replace('_', ' ')} is {item}.")


def extract_memory(user_input):
    """Ask Qwen3 to extract structured user-specific facts."""
    try:
        response = ollama.chat(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": '''You are a careful memory-extraction system.
Extract only useful facts explicitly stated about the user for future conversations.

Possible fields:
- name: string
- age: number
- education: string or list of strings
- learning: list of strings
- skills: list of strings
- programming_languages: list of strings
- projects: list of strings
- goals: list of strings
- preferences: list of strings
- hobbies: list of strings
- tools: list of strings (software, applications, IDEs, and developer tools)

Rules:
- For learning, skills, programming_languages, projects, goals, preferences,
  hobbies, and tools, ALWAYS return a JSON list, even for one item.
- Do not infer or invent facts.
- Do not save questions, temporary requests, or general knowledge.
- If the user mentions working on a project IN a programming language (e.g. "project in Java"), assign that language to 'programming_languages', NOT 'projects'. 'projects' is strictly for specific project names (e.g. "weather app", "portfolio website").
- Return ONLY valid JSON, with no Markdown fences.
- If there is no useful user-specific fact, return {}.

Examples:
User: My name is Aaryan.
Output: {"name": "Aaryan"}
User: I am 21 years old.
Output: {"age": 21}
User: I am learning C++.
Output: {"learning": ["C++"]}
User: I know Python and Java.
Output: {"programming_languages": ["Python", "Java"]}
User: I am making a project in Java.
Output: {"programming_languages": ["Java"]}
User: I am building a portfolio website in React.
Output: {"projects": ["portfolio website"], "skills": ["React"]}
User: I use VS Code and Android Studio.
Output: {"tools": ["VS Code", "Android Studio"]}
User: What is gradient descent?
Output: {}''',
                },
                {"role": "user", "content": user_input},
            ],
        )

        text = response.get("message", {}).get("content", "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        data = json.loads(text)

        if not isinstance(data, dict):
            return {}

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
        return {
            key: value
            for key, value in data.items()
            if key in allowed_fields and value not in (None, "", [], {})
        }

    except Exception:
        return {}


def update_semantic_memory(user_input):
    """Extract new facts, convert them to atomic statements, embed and index them into FAISS."""
    extracted = extract_memory(user_input)
    if not extracted:
        return {}

    newly_added = {}
    new_statements = []

    existing_statements = {item["statement"].casefold() for item in memory_facts}

    for key, value in extracted.items():
        items = value if isinstance(value, list) else [value]
        added_for_key = []

        for item in items:
            if not str(item).strip():
                continue
            statement = fact_to_statement(key, item)
            if statement.casefold() not in existing_statements:
                fact_record = {
                    "id": len(memory_facts) + len(new_statements),
                    "category": key,
                    "item": item,
                    "statement": statement,
                }
                new_statements.append(fact_record)
                existing_statements.add(statement.casefold())
                added_for_key.append(item)

        if added_for_key:
            newly_added[key] = added_for_key

    if new_statements:
        vectors = [get_embedding(record["statement"]) for record in new_statements]
        stacked = np.vstack(vectors).astype("float32")
        memory_index.add(stacked)
        memory_facts.extend(new_statements)
        save_memory_store()

        print("\nSemantic Memory updated with new atomic facts:")
        for record in new_statements:
            print(f"- [{record['category']}] {record['statement']}")

    return newly_added


def retrieve_relevant_memories(query, top_k=TOP_K_MEMORY, threshold=MEMORY_SIMILARITY_THRESHOLD):
    """
    Search memory FAISS index using cosine similarity.
    Filter out any results below the threshold to prevent context pollution.
    """
    if memory_index is None or not memory_facts or memory_index.ntotal == 0:
        return []

    query_vec = get_embedding(query)
    search_k = min(top_k, memory_index.ntotal)
    scores, indices = memory_index.search(query_vec, search_k)

    relevant = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(memory_facts):
            continue
        if score >= threshold:
            fact = memory_facts[idx]
            relevant.append({
                "statement": fact["statement"],
                "category": fact["category"],
                "score": float(score),
            })

    return relevant


def show_memory():
    if not memory_facts:
        print("Semantic memory is empty.")
        return
    print(f"\n--- SEMANTIC LONG-TERM MEMORY ({len(memory_facts)} facts indexed) ---")
    for fact in memory_facts:
        print(f"[{fact['id']}] ({fact['category']}): {fact['statement']}")


def search_memory_command(query):
    """Directly test semantic memory search and inspect similarity scores."""
    print(f"\nSearching semantic memory for: '{query}'")
    results = retrieve_relevant_memories(query, top_k=5, threshold=0.0)  # Show all scores for education
    if not results:
        print("No memories stored.")
        return
    print(f"{'Score':<8} | {'Status':<10} | Statement")
    print("-" * 65)
    for r in results:
        status = "ACCEPTED" if r["score"] >= MEMORY_SIMILARITY_THRESHOLD else "FILTERED"
        print(f"{r['score']:.4f}   | {status:<10} | {r['statement']}")


def clear_memory():
    global memory_index, memory_facts
    memory_facts = []
    memory_index = faiss.IndexFlatIP(MEMORY_EMBED_DIM)
    save_memory_store()
    print("Semantic memory cleared.")


# ---------- Deterministic Direct Fast-Path ----------

def retrieve_deterministic_answer(user_input):
    """Fast-path lookup for explicit identity queries."""
    text = user_input.casefold().strip()
    text = re.sub(r"[?!.,]+$", "", text).strip()

    if text in {"who am i", "what do you know about me", "what do you remember about me", "show my memories"}:
        if not memory_facts:
            return "I don't have any long-term memories about you yet."
        facts = [f"- {fact['statement']}" for fact in memory_facts]
        return "Here is what I have saved about you in semantic memory:\n" + "\n".join(facts)

    return None


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
    if name in {"weather", "calculator"}:
        return "VALID"
    if name == "search_knowledge":
        if "No relevant information found" in result or "unavailable" in result:
            return "RETRY"
    return "VALID"


def extract_math_expression(user_input):
    text = user_input.lower().strip()
    for phrase in ["what is", "calculate", "solve"]:
        text = text.replace(phrase, "")
    text = text.replace("?", "").strip()
    if re.fullmatch(r"[0-9+\-*/(). %]+", text):
        return text
    return None


# ---------- Agent Loop ----------

conversation_history = []


def run_agent(user_input):
    # 1. Deterministic direct check for identity
    direct_answer = retrieve_deterministic_answer(user_input)
    if direct_answer is not None:
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": direct_answer})
        return direct_answer

    # 2. Math router
    math_expression = extract_math_expression(user_input)
    if math_expression:
        print("\nRouter: calculator")
        result = calculator(math_expression)
        print(f"Result: {result}")
        answer = f"Calculation failed: {result}" if result.startswith("ERROR:") else f"The result is {result}."
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})
        return answer

    # 3. Update semantic memory with any facts from this turn
    newly_added = update_semantic_memory(user_input)

    # 4. Semantic Memory Retrieval: retrieve only relevant facts for THIS specific turn
    relevant_memories = retrieve_relevant_memories(user_input)

    memory_context = ""
    if relevant_memories:
        print(f"\n[Semantic Memory] Retrieved {len(relevant_memories)} relevant facts (Threshold >= {MEMORY_SIMILARITY_THRESHOLD}):")
        for mem in relevant_memories:
            print(f"  • ({mem['score']:.3f}) {mem['statement']}")
        facts_text = "\n".join(f"- {mem['statement']}" for mem in relevant_memories)
        memory_context = f"\n\nRelevant user facts retrieved from semantic memory:\n{facts_text}"
    else:
        print("\n[Semantic Memory] No stored facts were sufficiently relevant to this query. Zero memory injected.")

    new_facts_note = ""
    if newly_added:
        new_facts_note = (
            f"\n\nNOTICE: The user just shared new facts in this turn: {json.dumps(newly_added, ensure_ascii=False)}. "
            "Acknowledge them naturally and confirm what was just saved."
        )

    system_prompt = f"""You are an AI agent.
Use the retrieved user memories when relevant to personalize your response, but do not invent facts.
Use search_knowledge for questions about the local knowledge base.
Use calculator for mathematical calculations.
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
    print("=" * 70)
    print("AI Agent V12 (Semantic Long-Term Memory) started.")
    print("Commands:")
    print("  /memory              - List all atomic facts in memory store")
    print("  /search_memory <txt> - Test semantic search & see cosine scores")
    print("  /clear_memory        - Wipe long-term semantic memory")
    print("  /clear_history       - Wipe current session chat history")
    print("  exit / quit          - Exit agent")
    print("=" * 70)

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
            show_memory()
            continue
        if user_input.startswith("/search_memory"):
            query_text = user_input.replace("/search_memory", "", 1).strip()
            if query_text:
                search_memory_command(query_text)
            else:
                print("Usage: /search_memory <query text>")
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


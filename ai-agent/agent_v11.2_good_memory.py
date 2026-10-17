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
TOP_K = 3

STORE_DIR = "vector_store"
INDEX_FILE = os.path.join(STORE_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
MEMORY_FILE = "memory.json"

LIST_FIELDS = {
    "learning",
    "skills",
    "projects",
    "goals",
    "preferences",
    "hobbies",
    "programming_languages",
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
}


# ---------- Memory ----------

def load_memory():
    """Load saved memory. Return an empty dictionary if it is missing/invalid."""
    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
        print("Warning: memory.json must contain a JSON object. Starting with empty memory.")
    except (json.JSONDecodeError, OSError) as error:
        print(f"Warning: could not load memory.json: {error}")

    return {}


def save_memory(memory_data):
    """Save memory using a temporary file to reduce the risk of partial writes."""
    temporary_file = MEMORY_FILE + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(memory_data, file, indent=4, ensure_ascii=False)
    os.replace(temporary_file, MEMORY_FILE)


memory = load_memory()


def show_memory():
    if not memory:
        print("Memory is empty.")
        return

    print("\n--- LONG-TERM MEMORY ---")
    for key, value in memory.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


def clear_memory():
    memory.clear()
    save_memory(memory)
    print("Long-term memory cleared.")


def extract_memory(user_input):
    """Ask Qwen3 to extract user-specific facts as a JSON object."""
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

Rules:
- For learning, skills, programming_languages, projects, goals, preferences,
  and hobbies, ALWAYS return a JSON list, even for one item.
- Do not infer or invent facts.
- Do not save questions, temporary requests, or general knowledge.
- Return ONLY valid JSON, with no Markdown fences.
- If there is no useful user-specific fact, return {}.

Examples:
User: My name is Aaryan.
Output: {"name": "Aaryan"}
User: I am 21 years old.
Output: {"age": 21}
User: I am learning C++.
Output: {"learning": ["C++"]}
User: I am also learning Python.
Output: {"learning": ["Python"]}
User: I know Python and Java.
Output: {"programming_languages": ["Python", "Java"]}
User: What is overfitting?
Output: {}'''
                },
                {"role": "user", "content": user_input},
            ],
        )

        text = response.get("message", {}).get("content", "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        data = json.loads(text)

        if not isinstance(data, dict):
            return {}

        # Basic validation: allow only known memory fields.
        allowed_fields = set(MEMORY_LABELS)
        return {
            key: value
            for key, value in data.items()
            if key in allowed_fields and value not in (None, "", [], {})
        }

    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        print(f"Memory extraction returned invalid JSON/data: {error}")
        return {}
    except Exception as error:
        print(f"Memory extraction failed: {error}")
        return {}


def update_memory(user_input):
    """Merge list-like facts and update single-value facts."""
    extracted = extract_memory(user_input)

    if not extracted:
        return False

    changed = False

    for key, value in extracted.items():
        if key in LIST_FIELDS:
            if isinstance(value, str):
                new_values = [value]
            elif isinstance(value, list):
                new_values = value
            else:
                continue

            old_values = memory.get(key, [])
            if isinstance(old_values, str):
                old_values = [old_values]  # migrate old V11 scalar values
            elif not isinstance(old_values, list):
                old_values = []

            merged = list(old_values)
            for item in new_values:
                if not isinstance(item, str) or not item.strip():
                    continue
                item = item.strip()
                duplicate = any(
                    isinstance(old, str) and old.strip().casefold() == item.casefold()
                    for old in merged
                )
                if not duplicate:
                    merged.append(item)
                    changed = True

            if merged != memory.get(key):
                memory[key] = merged
                changed = True

        else:
            # Single-value fields (such as name/age) use the newest extracted value.
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                if memory.get(key) != value:
                    memory[key] = value
                    changed = True

    if changed:
        save_memory(memory)
        print("\nMemory updated:")
        for key in extracted:
            print(f"{key}: {memory.get(key)}")
        return True

    return False



def retrieve_memory_answer(user_input):
    """Answer clear memory questions without treating statements as questions."""
    text = user_input.casefold().strip()
    text = re.sub(r"[?!.,]+$", "", text).strip()

    # Recognize explicit questions about the user's identity.
    if text in {"who am i", "who am i?"}:
        if "name" in memory:
            return f"Your name is {memory['name']}."
        return "I don't have your name saved in long-term memory yet."

    # Recognize requests to summarize stored facts.
    summary_questions = {
        "what do you know about me",
        "what do you remember about me",
        "show my memories",
        "show me what you remember about me",
    }

    if text in summary_questions:
        if not memory:
            return "I don't have any long-term memories about you yet."

        facts = [
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in memory.items()
        ]
        return "Here's what I remember:\n- " + "\n- ".join(facts)

    # Only route actual questions, not statements such as
    # "My programming languages are Python and C++."
    question_starters = (
        "what is my ",
        "what are my ",
        "what's my ",
        "which ",
        "do you remember ",
        "tell me my ",
        "what am i ",
        "what do i ",
        "which programming languages ",
        "which languages ",
    )

    is_question = (
        text.startswith(question_starters)
        or text.startswith("what is my ")
        or text.startswith("what are my ")
    )

    if not is_question:
        return None

    aliases = {
        "programming_languages": [
            "programming languages",
            "coding languages",
            "languages",
        ],
        "learning": ["learning", "learn"],
        "education": ["education", "degree", "studying"],
        "projects": ["projects", "project"],
        "skills": ["skills", "skill"],
        "goals": ["goals", "goal"],
        "preferences": ["preferences", "prefer"],
        "hobbies": ["hobbies", "hobby", "interests"],
        "age": ["age", "old"],
        "name": ["name", "called"],
    }

    # More specific fields must be checked before broad aliases.
    if "learn" in text:
        ordered_keys = ["learning"] + [
            key for key in aliases if key != "learning"
        ]
    else:
        ordered_keys = sorted(
            aliases,
            key=lambda key: max(len(alias) for alias in aliases[key]),
            reverse=True,
        )

    for key in ordered_keys:
        if any(alias in text for alias in aliases[key]):
            if key not in memory:
                label = MEMORY_LABELS.get(key, key.replace("_", " "))
                return f"I don't have your {label} saved in long-term memory yet."

            value = memory[key]
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)

            label = MEMORY_LABELS.get(key, key.replace("_", " "))
            return f"Your {label}: {value}."

    return None


# ---------- Embeddings ----------

def get_embedding(text):
    response = ollama.embed(model="nomic-embed-text", input=text)
    return np.array(response["embeddings"][0], dtype="float32")


# ---------- Load vector store ----------

def load_vector_store():
    """Load FAISS files if present; allow the agent to run without a vector store."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print("Vector store not found. Knowledge search will be unavailable until it is built.")
        return None, []

    try:
        loaded_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            loaded_chunks = json.load(file)
        if not isinstance(loaded_chunks, list):
            raise ValueError("chunks.json must contain a list")
        return loaded_index, loaded_chunks
    except Exception as error:
        print(f"Could not load vector store: {error}")
        return None, []


index, chunks = load_vector_store()


# ---------- Knowledge search ----------

def search_knowledge(query):
    if index is None or not chunks:
        return "Knowledge search is unavailable because the vector store is missing or empty."

    query_embedding = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, min(TOP_K, len(chunks)))

    results = []
    for index_id in indices[0]:
        if index_id == -1 or index_id >= len(chunks):
            continue
        chunk = chunks[index_id]
        results.append(f"Source: {chunk.get('source', 'Unknown')}\n{chunk.get('text', '')}")

    return "\n\n".join(results) if results else "No relevant information found."


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
        # Avoid pathological exponentiation consuming excessive resources.
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
    # This is placeholder data, not live weather information.
    return f"Weather lookup is not connected to a live weather service for {city}."


tools = {
    "calculator": calculator,
    "weather": weather,
    "search_knowledge": search_knowledge,
}


# ---------- Tool definitions ----------

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


# ---------- Tool execution ----------

def execute_tool(name, arguments):
    if name not in tools:
        return f"ERROR: Tool '{name}' does not exist."
    try:
        return str(tools[name](**arguments))
    except Exception as error:
        return f"ERROR: {error}"


# ---------- Verification ----------

def verify_tool_result(name, result):
    if result.startswith("ERROR:"):
        return "RETRY"

    # A weather placeholder/unavailable response must not be represented as live data.
    if name == "weather":
        return "VALID"
    if name == "calculator":
        return "VALID"

    if name == "search_knowledge":
        if "No relevant information found" in result or "unavailable" in result:
            return "RETRY"
        try:
            response = ollama.chat(
                model="qwen3",
                messages=[
                    {
                        "role": "system",
                        "content": "Decide whether retrieved text contains potentially relevant information. Reply only VALID or RETRY.",
                    },
                    {
                        "role": "user",
                        "content": f"User query context is not included here. Retrieved text:\n{result}\n\nIs it potentially relevant? Reply only VALID or RETRY.",
                    },
                ],
            )
            decision = response.get("message", {}).get("content", "").strip().upper()
            return "RETRY" if decision.startswith("RETRY") else "VALID"
        except Exception:
            # Don't crash the agent if the verifier model call fails.
            return "VALID"

    return "VALID"


# ---------- Math router ----------

def extract_math_expression(user_input):
    text = user_input.lower().strip()
    for phrase in ["what is", "calculate", "solve"]:
        text = text.replace(phrase, "")
    text = text.replace("?", "").strip()

    if re.fullmatch(r"[0-9+\-*/(). %]+", text):
        return text
    return None


# ---------- Agent ----------

def run_agent(user_input):
    # Answer recognized memory questions directly.
    memory_answer = retrieve_memory_answer(user_input)
    if memory_answer is not None:
        return memory_answer

    # Extract and save facts from ordinary user messages.
    update_memory(user_input)

    math_expression = extract_math_expression(user_input)
    if math_expression:
        print("\nRouter: calculator")
        result = calculator(math_expression)
        print(f"Result: {result}")
        if result.startswith("ERROR:"):
            return f"Calculation failed: {result}"
        return f"The result is {result}."

    # Keep the rest of your existing run_agent() unchanged.

    memory_context = ""
    if memory:
        memory_context = "\n\nLong-term memory (stored user facts):\n" + json.dumps(memory, indent=2, ensure_ascii=False)

    messages = [
        {
            "role": "system",
            "content": """You are an AI agent.
Use the user's long-term memory when relevant, but do not invent facts.
Use search_knowledge for questions about the local knowledge base.
Use calculator for mathematical calculations.
For weather, clearly state that live weather is unavailable unless a live service is connected.
If the stored memory does not contain a requested personal fact, say you do not have it saved.
""" + memory_context,
        },
        {"role": "user", "content": user_input},
    ]

    retries = 0
    for _ in range(MAX_ITERATIONS):
        try:
            response = ollama.chat(model="qwen3", messages=messages, tools=tool_definitions)
        except Exception as error:
            return f"Could not contact Ollama. Check that Ollama is running and Qwen3 is available. Details: {error}"

        message = response.get("message", {})
        messages.append(message)
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return message.get("content", "")

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
                    return "Tool failed after several attempts."
                messages.append({
                    "role": "user",
                    "content": f"The tool result was not useful: {result}. Try a different approach or explain the limitation.",
                })
                continue

            messages.append({"role": "tool", "tool_name": name, "content": result})

    return "Maximum iterations reached."


# ---------- Main ----------

if __name__ == "__main__":
    print("AI Agent V11.2 started. Type /memory to inspect memory, /clear_memory to clear it, or exit to quit.")

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
        if user_input == "/clear_memory":
            clear_memory()
            continue

        answer = run_agent(user_input)
        print("\nAgent:", answer)

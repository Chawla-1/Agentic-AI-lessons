import os
import json
import re
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


# ---------- Memory ----------

def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(memory):

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)


memory = load_memory()


def show_memory():

    if not memory:
        print("Memory is empty.")
        return

    print("\n--- LONG TERM MEMORY ---")

    for key, value in memory.items():
        print(f"{key}: {value}")


def clear_memory():

    memory.clear()
    save_memory(memory)

    print("Long-term memory cleared.")


# ---------- Intelligent memory extraction ----------

def extract_memory(user_input):

    response = ollama.chat(
        model="qwen3",
        messages=[
            {
                "role": "system",
                "content": """
You are a memory extraction system.

Extract information that is useful to remember
about the user for future conversations.

Remember things such as:
- name
- age
- education
- skills
- programming languages
- projects
- goals
- preferences
- hobbies

Do NOT remember:
- questions
- temporary requests
- general facts
- information unrelated to the user

Return ONLY valid JSON.

If there is nothing worth remembering, return:

{}

Example:

User: My name is Aaryan.
Output:
{"name": "Aaryan"}

User: I am 21 years old.
Output:
{"age": 21}

User: I am learning C++.
Output:
{"learning": "C++"}

User: What is overfitting?
Output:
{}
"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    )

    text = response["message"]["content"].strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    return {}


def update_memory(user_input):

    extracted = extract_memory(user_input)

    if not extracted:
        return False

    memory.update(extracted)

    save_memory(memory)

    print("\nMemory updated:")

    for key, value in extracted.items():
        print(f"{key}: {value}")

    return True


# ---------- Embeddings ----------

def get_embedding(text):

    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return np.array(
        response["embeddings"][0],
        dtype="float32"
    )


# ---------- Load vector store ----------

def load_vector_store():

    if not os.path.exists(INDEX_FILE):
        print("Vector store not found.")
        exit()

    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return index, chunks


index, chunks = load_vector_store()


# ---------- Knowledge search ----------

def search_knowledge(query):

    query_embedding = get_embedding(query)
    query_embedding = query_embedding.reshape(1, -1)

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    results = []

    for index_id in indices[0]:

        if index_id == -1:
            continue

        chunk = chunks[index_id]

        results.append(
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    if not results:
        return "No relevant information found."

    return "\n\n".join(results)


# ---------- Calculator ----------

def calculator(expression):

    allowed = set("0123456789+-*/(). ")

    if not all(c in allowed for c in expression):
        return "ERROR: Invalid expression."

    try:
        return str(eval(expression))

    except Exception as e:
        return f"ERROR: {e}"


# ---------- Weather ----------

def weather(city):

    return f"The weather in {city} is 28°C and cloudy."


tools = {
    "calculator": calculator,
    "weather": weather,
    "search_knowledge": search_knowledge
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
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Gets weather information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Searches the local knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ---------- Tool execution ----------

def execute_tool(name, arguments):

    if name not in tools:
        return f"ERROR: Tool '{name}' does not exist."

    try:
        return str(tools[name](**arguments))

    except Exception as e:
        return f"ERROR: {e}"


# ---------- Verification ----------

def verify_tool_result(name, result):

    if result.startswith("ERROR:"):
        return "RETRY"

    if name in ["calculator", "weather"]:
        return "VALID"

    if name == "search_knowledge":

        response = ollama.chat(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Verify retrieved knowledge. "
                        "Reply only VALID or RETRY."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Knowledge:\n{result}\n\n"
                        "Is it useful? "
                        "Reply only VALID or RETRY."
                    )
                }
            ]
        )

        decision = response["message"]["content"].strip().upper()

        return "RETRY" if "RETRY" in decision else "VALID"

    return "VALID"


# ---------- Math router ----------

def extract_math_expression(user_input):

    text = user_input.lower().strip()

    for word in ["what is", "calculate", "solve"]:
        text = text.replace(word, "")

    text = text.replace("?", "").strip()

    if re.fullmatch(r"[0-9+\-*/(). ]+", text):
        return text

    return None


# ---------- Agent ----------

def run_agent(user_input):

    update_memory(user_input)

    if user_input.lower() == "what do you know about me":

        if not memory:
            return "I don't have any long-term memories about you."

        facts = []

        for key, value in memory.items():
            facts.append(
                f"{key.replace('_', ' ').title()}: {value}"
            )

        return "Here's what I remember:\n- " + "\n- ".join(facts)

    math_expression = extract_math_expression(user_input)

    if math_expression:

        print("\nRouter: calculator")

        result = calculator(math_expression)

        print(f"Result: {result}")

        if result.startswith("ERROR:"):
            return f"Calculation failed: {result}"

        return f"The result is {result}."

    memory_context = ""

    if memory:

        memory_context = (
            "\n\nLong-term memory:\n"
            + json.dumps(memory, indent=2)
        )

    messages = [
        {
            "role": "system",
            "content": """
You are an AI agent.

Use the user's long-term memory when relevant.

Use search_knowledge for local knowledge.

Use calculator for mathematical calculations.

Use weather for weather requests.

Do not invent memories.
"""
            + memory_context
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    retries = 0

    for _ in range(MAX_ITERATIONS):

        response = ollama.chat(
            model="qwen3",
            messages=messages,
            tools=tool_definitions
        )

        message = response["message"]

        messages.append(message)

        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return message.get("content", "")

        for call in tool_calls:

            name = call["function"]["name"]
            arguments = call["function"]["arguments"]

            print(f"\nTool: {name}")
            print(f"Arguments: {arguments}")

            result = execute_tool(
                name,
                arguments
            )

            print(f"Result: {result}")

            decision = verify_tool_result(
                name,
                result
            )

            print(f"Verification: {decision}")

            if decision == "RETRY":

                retries += 1

                if retries > MAX_RETRIES:
                    return "Tool failed after several attempts."

                messages.append({
                    "role": "user",
                    "content": "The tool failed. Try another approach."
                })

                continue

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": result
            })

    return "Maximum iterations reached."


# ---------- Main ----------

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    if user_input.lower() == "/memory":

        show_memory()
        continue

    if user_input.lower() == "/clear_memory":

        clear_memory()
        continue

    answer = run_agent(user_input)

    print("\nAgent:", answer)
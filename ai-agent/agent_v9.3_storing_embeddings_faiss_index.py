import os
import json
import ollama
import faiss
import numpy as np


MAX_ITERATIONS = 10
TOP_K = 3

STORE_DIR = "vector_store"
INDEX_FILE = os.path.join(STORE_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")


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


# ---------- Build vector store ----------

def build_vector_store():

    os.makedirs(STORE_DIR, exist_ok=True)

    chunks = []

    for filename in os.listdir("knowledge"):

        path = os.path.join("knowledge", filename)

        if not filename.endswith(".txt"):
            continue

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        for paragraph in paragraphs:

            chunks.append({
                "source": filename,
                "text": paragraph
            })

    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = [
        get_embedding(chunk["text"])
        for chunk in chunks
    ]

    matrix = np.array(embeddings, dtype="float32")

    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])

    index.add(matrix)

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2)

    print("Vector store created.")


# ---------- Load vector store ----------

def load_vector_store():

    if not os.path.exists(INDEX_FILE):
        build_vector_store()

    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks.")

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

    for score, index_id in zip(scores[0], indices[0]):

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


# ---------- Other tools ----------

def calculator(expression):

    allowed = set("0123456789+-*/(). ")

    if not all(c in allowed for c in expression):
        return "Invalid expression."

    try:
        return str(eval(expression))
    except:
        return "Could not calculate."


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
            "description": "Gets weather information for a city.",
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
            "description": (
                "Searches the local knowledge base. "
                "Use this for questions about the "
                "knowledge stored in the local files."
            ),
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
        return f"Tool '{name}' does not exist."

    try:
        return str(tools[name](**arguments))

    except Exception as e:
        return f"Tool error: {e}"


# ---------- Agent ----------

def run_agent(user_input):

    messages = [
        {
            "role": "system",
            "content": """
You are an AI agent.

You have access to a local knowledge base.

For questions about machine learning, Python,
neural networks, or information that may exist
in the local knowledge base, use search_knowledge
before answering.

Use calculator for mathematical calculations.

Use weather for weather requests.

Do not invent information from the knowledge base.
"""
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

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

            print(f"Result:\n{result}")

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

    answer = run_agent(user_input)

    print("\nAgent:", answer)
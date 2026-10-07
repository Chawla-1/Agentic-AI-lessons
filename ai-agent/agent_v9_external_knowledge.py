import os
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_ITERATIONS = 10
TOP_K = 3


# ---------- Load and chunk knowledge ----------

chunks = []

for filename in os.listdir("knowledge"):

    path = os.path.join("knowledge", filename)

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


texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(texts)


# ---------- Tools ----------

def calculator(expression: str) -> str:

    allowed = set("0123456789+-*/(). ")

    if not all(c in allowed for c in expression):
        return "Invalid expression."

    try:
        return str(eval(expression))
    except:
        return "Could not calculate."


def weather(city: str) -> str:

    return f"The weather in {city} is 28°C and cloudy."


def search_knowledge(query: str) -> str:

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        vectors
    )[0]

    top_indices = scores.argsort()[-TOP_K:][::-1]

    results = []

    for index in top_indices:

        if scores[index] < 0.05:
            continue

        chunk = chunks[index]

        results.append(
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    if not results:
        return "No relevant information found."

    return "\n\n".join(results)


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
                    "expression": {"type": "string"}
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
                    "city": {"type": "string"}
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
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]


# ---------- Execute tool ----------

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

Use search_knowledge when information from the
local knowledge base is needed.

Use calculator for calculations.

Use weather for weather requests.

Only use information returned by tools when
answering knowledge-base questions.
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
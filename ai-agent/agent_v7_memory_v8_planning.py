import json
import ollama

MAX_ITERATIONS = 10


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


tools = {
    "calculator": calculator,
    "weather": weather
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
    }
]


# ---------- Agent state ----------

state = {
    "messages": [],
    "plan": [],
    "completed_steps": [],
    "tool_history": [],
    "iterations": 0
}


# ---------- Create plan ----------

def create_plan(user_input):

    prompt = f"""
        Create a simple plan for this task:

        {user_input}

        Return ONLY a JSON array.

        Example:
        ["Get weather in Delhi", "Calculate 25 * 30"]
        """

    response = ollama.chat(
        model="qwen3",
        messages=[
            {
                "role": "system",
                "content": "You are a task planner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response["message"]["content"]

    try:
        return json.loads(text)
    except:
        return [user_input]


# ---------- Execute tool ----------

def execute_tool(name, arguments):

    if name not in tools:
        return f"Tool '{name}' does not exist."

    try:
        result = tools[name](**arguments)

        state["tool_history"].append({
            "tool": name,
            "arguments": arguments,
            "result": result
        })

        return str(result)

    except Exception as e:
        return f"Tool error: {e}"


# ---------- Run agent ----------

def run_agent(user_input):

    state["plan"] = create_plan(user_input)
    state["completed_steps"] = []
    state["tool_history"] = []
    state["iterations"] = 0

    print("\nPlan:")

    for i, step in enumerate(state["plan"], 1):
        print(f"{i}. {step}")

    state["messages"] = [
        {
            "role": "system",
            "content": """
You are an AI agent.

Complete the user's task using the available tools.

Follow the provided plan.

Use tools when necessary.

After completing the task, provide a concise final answer.
"""
        },
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "system",
            "content": f"Plan: {state['plan']}"
        }
    ]

    while state["iterations"] < MAX_ITERATIONS:

        state["iterations"] += 1

        response = ollama.chat(
            model="qwen3",
            messages=state["messages"],
            tools=tool_definitions
        )

        message = response["message"]

        state["messages"].append(message)

        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return message.get("content", "")

        for call in tool_calls:

            name = call["function"]["name"]
            arguments = call["function"]["arguments"]

            print(f"\nUsing tool: {name}")
            print(f"Arguments: {arguments}")

            result = execute_tool(name, arguments)

            print(f"Result: {result}")

            state["completed_steps"].append({
                "tool": name,
                "result": result
            })

            state["messages"].append({
                "role": "tool",
                "tool_name": name,
                "content": result
            })

    return "Agent stopped: maximum iterations reached."


# ---------- Main ----------

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    if user_input.lower() == "/reset":
        state = {
            "messages": [],
            "plan": [],
            "completed_steps": [],
            "tool_history": [],
            "iterations": 0
        }

        print("State cleared.")
        continue

    if user_input.lower() == "/plan":
        print(json.dumps(state["plan"], indent=2))
        continue

    if user_input.lower() == "/state":
        print(json.dumps(state, indent=2, default=str))
        continue

    answer = run_agent(user_input)

    print("\nAgent:", answer)
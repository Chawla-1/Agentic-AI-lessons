import json


def calculator(expression):
    return eval(expression)


def weather(city):
    return f"The weather in {city} is 28°C and cloudy."


# All available tools
tools = {
    "calculator": calculator,
    "weather": weather
}


# -------------------------
# LLM DECISION (SIMULATED)
# -------------------------

def llm_decide(user_input):

    text = user_input.lower()

    if "calculate" in text or "multiply" in text or "add" in text:
        
        expression = input("What expression should I calculate? ")

        return {
            "tool": "calculator",
            "arguments": {
                "expression": expression
            }
        }

    elif "weather" in text:

        city = input("Which city? ")

        return {
            "tool": "weather",
            "arguments": {
                "city": city
            }
        }

    else:

        return {
            "tool": None,
            "arguments": {}
        }


# -------------------------
# AGENT
# -------------------------

def agent(user_input):

    # Ask the "LLM" what action to take
    decision = llm_decide(user_input)

    print("\nDecision:")
    print(json.dumps(decision, indent=4))

    # No tool required
    if decision["tool"] is None:
        return "I don't need a tool for this request."

    # Get selected tool
    tool_name = decision["tool"]

    if tool_name not in tools:
        return f"Unknown tool: {tool_name}"

    tool = tools[tool_name]

    # Get arguments
    arguments = decision["arguments"]

    # Execute tool
    result = tool(**arguments)

    return result


# -------------------------
# MAIN LOOP
# -------------------------

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = agent(user_input)

    print("\nAgent:", result)
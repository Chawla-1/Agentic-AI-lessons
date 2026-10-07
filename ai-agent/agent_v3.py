import json


# ==========================================
# TOOLS
# ==========================================

def calculator(expression):
    return eval(expression)


def weather(city):
    return f"The weather in {city} is 28°C and cloudy."


tools = {
    "calculator": calculator,
    "weather": weather
}


# ==========================================
# SIMULATED LLM
# ==========================================

def llm_decide(user_input):

    text = user_input.lower()

    # --------------------------
    # Calculator
    # --------------------------

    if "multiply" in text:

        numbers = [int(word) for word in text.split() if word.isdigit()]

        if len(numbers) >= 2:

            expression = f"{numbers[0]} * {numbers[1]}"

            return {
                "tool": "calculator",
                "arguments": {
                    "expression": expression
                }
            }

    elif "add" in text:

        numbers = [int(word) for word in text.split() if word.isdigit()]

        if len(numbers) >= 2:

            expression = f"{numbers[0]} + {numbers[1]}"

            return {
                "tool": "calculator",
                "arguments": {
                    "expression": expression
                }
            }

    # --------------------------
    # Weather
    # --------------------------

    elif "weather" in text:

        words = text.split()

        cities = ["delhi", "mumbai", "london", "paris"]

        for city in cities:

            if city in words:

                return {
                    "tool": "weather",
                    "arguments": {
                        "city": city.title()
                    }
                }

    # --------------------------
    # No tool
    # --------------------------

    return {
        "tool": None,
        "arguments": {}
    }


# ==========================================
# AGENT
# ==========================================

def agent(user_input):

    print("\n[1] User request:")
    print(user_input)

    # Ask LLM what to do
    decision = llm_decide(user_input)

    print("\n[2] Tool decision:")
    print(json.dumps(decision, indent=4))

    # No tool required
    if decision["tool"] is None:
        return "I don't need a tool for this request."

    # Find tool
    tool_name = decision["tool"]

    if tool_name not in tools:
        return f"Unknown tool: {tool_name}"

    tool = tools[tool_name]

    # Execute tool
    arguments = decision["arguments"]

    print("\n[3] Executing tool:")
    print(tool_name)

    result = tool(**arguments)

    print("\n[4] Tool result:")
    print(result)

    return result


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = agent(user_input)

    print("\nAgent:", response)
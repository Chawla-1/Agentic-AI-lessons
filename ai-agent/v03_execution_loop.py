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

def llm_decide(user_input, state):

    text = user_input.lower()

    # --------------------------------------
    # First decision: weather
    # --------------------------------------

    if "weather" in text and "weather" not in state["completed_tools"]:

        cities = ["delhi", "mumbai", "london", "paris"]

        for city in cities:
            if city in text:

                return {
                    "type": "tool_call",
                    "tool": "weather",
                    "arguments": {
                        "city": city.title()
                    }
                }

    # --------------------------------------
    # Second decision: calculation
    # --------------------------------------

    if (
        ("multiply" in text or "times" in text)
        and "calculator" not in state["completed_tools"]
    ):

        numbers = [
            int(word)
            for word in text.split()
            if word.isdigit()
        ]

        if len(numbers) >= 2:

            return {
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {
                    "expression": f"{numbers[0]} * {numbers[1]}"
                }
            }

    # --------------------------------------
    # Add operation
    # --------------------------------------

    if (
        "add" in text
        and "calculator" not in state["completed_tools"]
    ):

        numbers = [
            int(word)
            for word in text.split()
            if word.isdigit()
        ]

        if len(numbers) >= 2:

            return {
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {
                    "expression": f"{numbers[0]} + {numbers[1]}"
                }
            }

    # --------------------------------------
    # No more tools
    # --------------------------------------

    return {
        "type": "final"
    }


# ==========================================
# TOOL EXECUTOR
# ==========================================

def execute_tool(decision):

    tool_name = decision["tool"]
    arguments = decision["arguments"]

    if tool_name not in tools:
        return f"Unknown tool: {tool_name}"

    tool = tools[tool_name]

    result = tool(**arguments)

    return result


# ==========================================
# AGENT LOOP
# ==========================================

def agent(user_input):

    # --------------------------------------
    # STATE
    # --------------------------------------

    state = {
        "user_input": user_input,
        "completed_tools": [],
        "tool_results": []
    }

    print("\n==============================")
    print("AGENT STARTED")
    print("==============================")

    # --------------------------------------
    # AGENT LOOP
    # --------------------------------------

    while True:

        print("\n--- THINKING ---")

        decision = llm_decide(
            user_input,
            state
        )

        print("Decision:")
        print(json.dumps(decision, indent=4))

        # ----------------------------------
        # FINAL ANSWER
        # ----------------------------------

        if decision["type"] == "final":

            print("\n--- FINAL ---")

            if state["tool_results"]:

                return " and ".join(
                    str(result) for result in state["tool_results"]
                )

            return "I don't need any tools for this request."

        # ----------------------------------
        # TOOL CALL
        # ----------------------------------

        print("\n--- ACTION ---")

        tool_name = decision["tool"]

        print(f"Calling tool: {tool_name}")

        result = execute_tool(decision)

        print("\n--- OBSERVATION ---")

        print(result)

        # ----------------------------------
        # UPDATE STATE
        # ----------------------------------

        state["completed_tools"].append(
            tool_name
        )

        state["tool_results"].append(
            result
        )


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = agent(user_input)

    print("\nAgent:", response)
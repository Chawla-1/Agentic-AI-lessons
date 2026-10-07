import json


# ==========================================
# TOOLS
# ==========================================

def calculator(expression):
    """
    Safely evaluate basic mathematical expressions.
    """

    allowed = set("0123456789+-*/(). ")

    if not all(char in allowed for char in expression):
        return "Invalid mathematical expression."

    try:
        return eval(expression)
    except Exception:
        return "Could not calculate the expression."


def weather(city):
    return f"The weather in {city} is 28°C and cloudy."


# ==========================================
# TOOL REGISTRY
# ==========================================

tools = {

    "calculator": {
        "function": calculator,

        "description": "Performs mathematical calculations.",

        "parameters": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to calculate."
            }
        }
    },

    "weather": {
        "function": weather,

        "description": "Gets the current weather for a city.",

        "parameters": {
            "city": {
                "type": "string",
                "description": "Name of the city."
            }
        }
    }
}


# ==========================================
# DISPLAY AVAILABLE TOOLS
# ==========================================

def show_tools():

    print("\nAVAILABLE TOOLS:")

    for name, tool in tools.items():

        print(f"\n{name}")
        print(f"Description: {tool['description']}")
        print("Parameters:")

        for parameter, details in tool["parameters"].items():

            print(
                f"  - {parameter}: "
                f"{details['type']} - "
                f"{details['description']}"
            )


# ==========================================
# SIMULATED LLM
# ==========================================

def llm_decide(user_input, state):

    text = user_input.lower()

    # --------------------------------------
    # WEATHER
    # --------------------------------------

    if (
        "weather" in text
        and "weather" not in state["completed_tools"]
    ):

        cities = [
            "delhi",
            "mumbai",
            "london",
            "paris"
        ]

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
    # MULTIPLICATION
    # --------------------------------------

    if (
        (
            "multiply" in text
            or "times" in text
        )
        and "calculator"
        not in state["completed_tools"]
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
                    "expression":
                        f"{numbers[0]} * {numbers[1]}"
                }
            }

    # --------------------------------------
    # ADDITION
    # --------------------------------------

    if (
        "add" in text
        and "calculator"
        not in state["completed_tools"]
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
                    "expression":
                        f"{numbers[0]} + {numbers[1]}"
                }
            }

    # --------------------------------------
    # FINISH
    # --------------------------------------

    return {
        "type": "final"
    }


# ==========================================
# GENERIC TOOL EXECUTOR
# ==========================================

def execute_tool(decision):

    tool_name = decision["tool"]

    arguments = decision["arguments"]

    # Check whether tool exists

    if tool_name not in tools:

        return {
            "success": False,
            "error": f"Tool '{tool_name}' does not exist."
        }

    tool = tools[tool_name]

    function = tool["function"]

    # --------------------------------------
    # Validate arguments
    # --------------------------------------

    expected_parameters = tool["parameters"]

    for parameter in expected_parameters:

        if parameter not in arguments:

            return {
                "success": False,
                "error":
                    f"Missing argument: {parameter}"
            }

    # --------------------------------------
    # Execute
    # --------------------------------------

    try:

        result = function(**arguments)

        # Normalize result to string

        return {
            "success": True,
            "result": str(result)
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ==========================================
# AGENT
# ==========================================

def agent(user_input):

    state = {

        "user_input": user_input,

        "completed_tools": [],

        "tool_results": []
    }

    print("\n==============================")
    print("AGENT STARTED")
    print("==============================")

    show_tools()

    # ======================================
    # AGENT LOOP
    # ======================================

    while True:

        print("\n--- THINKING ---")

        decision = llm_decide(
            user_input,
            state
        )

        print(
            json.dumps(
                decision,
                indent=4
            )
        )

        # ----------------------------------
        # FINAL
        # ----------------------------------

        if decision["type"] == "final":

            print("\n--- FINAL ---")

            if not state["tool_results"]:

                return "No tools were required."

            return " | ".join(
                state["tool_results"]
            )

        # ----------------------------------
        # ACTION
        # ----------------------------------

        print("\n--- ACTION ---")

        print(
            f"Calling: {decision['tool']}"
        )

        # ----------------------------------
        # EXECUTE
        # ----------------------------------

        result = execute_tool(decision)

        print("\n--- OBSERVATION ---")

        print(
            json.dumps(
                result,
                indent=4
            )
        )

        # ----------------------------------
        # UPDATE STATE
        # ----------------------------------

        state["completed_tools"].append(
            decision["tool"]
        )

        if result["success"]:

            state["tool_results"].append(
                result["result"]
            )

        else:

            state["tool_results"].append(
                result["error"]
            )


# ==========================================
# MAIN
# ==========================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = agent(user_input)

    print("\nAgent:", response)
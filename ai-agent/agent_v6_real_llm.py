import ollama


# ============================================================
# 1. TOOLS
# ============================================================

def calculator(expression: str) -> str:
    """
    Perform a mathematical calculation.
    """

    # Temporary safety check.
    # We will replace eval() with a proper parser later.
    allowed = set("0123456789+-*/(). ")

    if not all(char in allowed for char in expression):
        return "Invalid mathematical expression."

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Could not calculate the expression."


def weather(city: str) -> str:
    # Fake weather for now.
    # Later we will connect a real API.

    return f"The weather in {city} is 28°C and cloudy."


# ============================================================
# 2. TOOL REGISTRY
# ============================================================

tools = {
    "calculator": calculator,
    "weather": weather
}


# ============================================================
# 3. TOOL DEFINITIONS GIVEN TO THE LLM
# ============================================================

tool_definitions = [

    {
        "type": "function",

        "function": {
            "name": "calculator",

            "description":
                "Performs mathematical calculations.",

            "parameters": {
                "type": "object",

                "properties": {

                    "expression": {
                        "type": "string",
                        "description":
                            "Mathematical expression to calculate."
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

            "description":
                "Gets weather information for a city.",

            "parameters": {
                "type": "object",

                "properties": {

                    "city": {
                        "type": "string",
                        "description":
                            "Name of the city."
                    }

                },

                "required": ["city"]
            }
        }
    }
]


# ============================================================
# 4. TOOL EXECUTOR
# ============================================================

def execute_tool(tool_name, arguments):

    print("\n================ TOOL CALL ================")

    print("Tool:")
    print(tool_name)

    print("\nArguments:")
    print(arguments)

    # Check whether tool exists
    if tool_name not in tools:

        return f"Tool '{tool_name}' does not exist."

    try:

        # Get actual Python function
        tool_function = tools[tool_name]

        # Execute it
        result = tool_function(**arguments)

        print("\nTool Result:")
        print(result)

        return str(result)

    except Exception as e:

        error = f"Tool execution failed: {e}"

        print(error)

        return error


# ============================================================
# 5. AGENT
# ============================================================

def run_agent(user_input):

    print("\n============================================")
    print("V6 AGENT")
    print("============================================")

    # --------------------------------------------------------
    # Conversation state
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",

            "content": """
                You are an AI agent.

                You have access to tools.

                Use a tool whenever it is necessary to answer the
                user's request.

                After receiving a tool result:

                - If another tool is needed, call it.
                - Otherwise, provide the final answer.

                Never invent tool results.
"""
        },

        {
            "role": "user",
            "content": user_input
        }

    ]

    # --------------------------------------------------------
    # AGENT LOOP
    # --------------------------------------------------------

    while True:

        print("\n--------------- LLM THINKING ---------------")

        # Ask the LLM what to do
        response = ollama.chat(

            model="qwen3",

            messages=messages,

            tools=tool_definitions
        )

        assistant_message = response["message"]

        # Save LLM response in state
        messages.append(assistant_message)

        # ----------------------------------------------------
        # CHECK FOR TOOL CALLS
        # ----------------------------------------------------

        tool_calls = assistant_message.get("tool_calls")

        if tool_calls:

            for call in tool_calls:

                # Extract tool name
                tool_name = call["function"]["name"]

                # Extract arguments
                arguments = call["function"]["arguments"]

                # Execute tool
                result = execute_tool(
                    tool_name,
                    arguments
                )

                # Send result back to LLM
                messages.append(
                    {
                        "role": "tool",

                        "tool_name": tool_name,

                        "content": result
                    }
                )

            # Go back to LLM
            continue

        # ----------------------------------------------------
        # NO TOOL CALL
        # Therefore this is the final answer
        # ----------------------------------------------------

        final_answer = assistant_message.get(
            "content",
            ""
        )

        print("\n================ FINAL ANSWER ================")

        print(final_answer)

        return final_answer


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:

            print("Agent stopped.")

            break

        run_agent(user_input)
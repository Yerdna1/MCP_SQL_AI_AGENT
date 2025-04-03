import asyncio
import sys
from app.graph.builder import compiled_graph # Import the compiled graph instance
from app.graph.state import AgentState # Import the correct state class
from app.config import settings # Import the settings instance
from langchain_core.messages import HumanMessage # Import HumanMessage

# Use the imported settings instance
config = settings

async def run_terminal_app():
    """Runs the application in terminal mode."""
    print("Starting SQL AI Agent in terminal mode...")

    # Use the imported compiled LangGraph agent
    app = compiled_graph

    print("Agent initialized. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nEnter your query: ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break

            if not user_input:
                continue

            # Prepare the initial state for the graph
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)], # Use HumanMessage
                # Config object 'settings' might be accessed directly within nodes if needed
                # Or pass specific settings if required by the initial state
                selected_llm_name='Ollama (Local)', # Use the name expected by the LLM provider
                agent_thoughts=[], # Initialize empty thoughts list
                # Add other necessary initial state fields if needed
                # e.g., db_schema=load_schema(...)
            )

            # Debug: Print the initial state before invoking
            print("\n--- Initial State ---")
            print(initial_state)
            print("---------------------")

            print("\nThinking...")
            # Invoke the graph asynchronously
            final_state = await app.ainvoke(initial_state)

            # Debug: Print the final state to understand its structure
            print("\n--- Final State ---")
            print(final_state)
            print("-------------------")

            # Display the final response from the agent
            if final_state and final_state.get("messages"):
                # Safely access the last message
                last_message = final_state["messages"][-1]
                print(f"\nDEBUG: Last message type: {type(last_message)}")
                print(f"DEBUG: Last message content: {last_message}")

                # Adapt based on observed structure (assuming Langchain Message objects)
                if hasattr(last_message, 'type') and hasattr(last_message, 'content'):
                    last_message_type = last_message.type
                    last_message_content = last_message.content
                    if last_message_type == "ai" or last_message_type == "assistant": # Handle common AI message types
                        print("\nResponse:")
                        print(last_message_content)
                    else:
                        print(f"\nFinal message type: {last_message_type}")
                        print(last_message_content)
                elif isinstance(last_message, tuple) and len(last_message) == 2:
                    # Handle the original tuple assumption if it works sometimes
                    last_message_type, last_message_content = last_message
                    if last_message_type == "assistant":
                        print("\nResponse:")
                        print(last_message_content)
                    else:
                        print(f"\nFinal message type: {last_message_type}")
                        print(last_message_content)
                else:
                    # Fallback if structure is unexpected
                    print("\nFinal message (raw):")
                    print(last_message)
            else:
                print("\nNo response generated.")

            # Optionally print intermediate steps or other state info
            # print("\n--- State ---")
            # print(final_state)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Optionally add more robust error handling or logging

if __name__ == "__main__":
    # Setup asyncio event loop based on OS
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(run_terminal_app())
    except Exception as e:
        print(f"Failed to run terminal application: {e}")

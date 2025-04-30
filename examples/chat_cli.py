import sys
import os
# ensure project root is on PYTHONPATH when running from examples/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import logging
from tinytroupe.agent import TinyPerson
from tinytroupe.hardcoded_personas import get_random_persona
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

# Set DEBUG flag for internal use
DEBUG = True

# Configure logging to write debug information to a file.
logging.basicConfig(
    filename="debug.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define persona colors.
PERSONA_COLORS = {
    "Oscar": "red",
    "Lisa": "green",
    "James": "magenta",
    "Sophia": "yellow",
    "Rahim": "cyan"
}
DEFAULT_PERSONA_COLOR = "white"

# Create a Rich Console instance for user-facing output.
console = Console()

def display_response(persona, response_output):
    """
    Displays the quick_talk response.
    The output is expected to be a dict with an 'action' field containing the 'talk' response.
    """
    content = response_output.get("action", {}).get("content", "")
    if not content.strip():
        console.print(f"[italic dim]{persona.name} returned an empty talk response.[/italic dim]")
    else:
        persona_color = PERSONA_COLORS.get(persona.name, DEFAULT_PERSONA_COLOR)
        console.print(
            Panel(
                content,
                title=f"[bold {persona_color}]{persona.name}[/bold {persona_color}]",
                border_style=persona_color,
            )
        )

def main():
    console.print("[bold]Welcome to the TinyTroupe Chat CLI![/bold]")
    console.print("Type 'exit' or 'quit' to leave.\n", style="dim")

    persona = get_random_persona()
    TinyPerson.communication_display = False  # Hide chain-of-thought.

    persona_color = PERSONA_COLORS.get(persona.name, DEFAULT_PERSONA_COLOR)
    logging.debug(f"Selected persona is {persona.name} with color {persona_color}")
    console.print(
        f"You are now chatting with [bold {persona_color}]{persona.name}[/bold {persona_color}]. Enjoy the conversation!\n"
    )

    while True:
        try:
            user_input = console.input("[blue]You:[/blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting the chat.", style="dim")
            break

        if user_input.lower() in ["exit", "quit"]:
            console.print("Exiting the chat.", style="dim")
            break

        # Append the new user input.
        persona.current_messages.append({"role": "user", "content": user_input})
        logging.debug("Current conversation history:")
        logging.debug(persona.current_messages)
        
        # Use quick_talk() to get a final 'talk' response that respects the persona.
        output = persona.quick_talk(max_content_length=1024)
        logging.debug("quick_talk raw output:")
        logging.debug(output)
        action_type = output.get("action", {}).get("type", None)
        if action_type != "talk":
            logging.debug(f"Expected action type 'talk', but got '{action_type}'")
        else:
            logging.debug("Talk action confirmed.")
        if output:
            reply = output.get("action", {}).get("content", "")
            logging.debug(f"Persona '{persona.name}' reply: {reply}")
        
        if output:
            display_response(persona, output)
        else:
            console.print(f"[red]{persona.name} provided no quick talk response.[/red]")
        
        # Clear out assistant responses so that only new user inputs remain.
        persona.current_messages = [msg for msg in persona.current_messages if msg["role"] == "user"]
        logging.debug("Conversation history after clearing assistant messages:")
        logging.debug(persona.current_messages)

if __name__ == "__main__":
    main()


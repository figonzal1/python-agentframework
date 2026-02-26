"""Shared terminal UI for agent demos using Rich."""
import asyncio
import json

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from agent_framework import Agent

console = Console()


def _format_tool_call(name: str, arguments: object) -> Spinner:
    """Formatea una llamada a tool con spinner y estilo de log."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            pass
    if isinstance(arguments, dict):
        args_str = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in arguments.items())
    else:
        args_str = str(arguments) if arguments else ""

    label = Text()
    label.append(f"{name}", style="grey46")
    label.append(f"({args_str})", style="grey42")

    return Spinner("dots", text=label, style="grey46")


async def chat(agent: Agent, user_input: str, title: str = "Agente") -> None:
    """Envía un mensaje al agente y muestra la respuesta con streaming."""
    console.print(Panel(user_input, title="[bold cyan]Tú[/bold cyan]", border_style="cyan"))

    collected = ""
    seen_tool_calls: set[str] = set()
    current_tool_renderable = None

    def renderable():
        """El contenido del Live cambia según el estado."""
        if collected:
            return Markdown(collected)
        if current_tool_renderable is not None:
            return current_tool_renderable
        return Spinner("dots", text="[bold yellow]Pensando...[/bold yellow]")

    stream = agent.run(user_input, stream=True)

    try:
        with Live(renderable(), console=console, refresh_per_second=12) as live:
            async for update in stream:
                # Detectar llamadas a tools — actualizar live sin interrumpirlo
                for content in update.contents or []:
                    if content.type == "function_call" and content.call_id not in seen_tool_calls:
                        seen_tool_calls.add(content.call_id)
                        current_tool_renderable = _format_tool_call(content.name, content.arguments)
                        live.update(renderable())

                # Acumular texto — reemplaza el tool call automáticamente
                if update.text:
                    collected += update.text
                    live.update(renderable())
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[dim]Respuesta interrumpida.[/dim]")

    if collected:
        console.print(Panel(Markdown(collected), title=f"[bold green]{title}[/bold green]", border_style="green"))


async def run_chat_loop(agent: Agent, title: str = "Agente con Ollama", agent_title: str = "Agente") -> None:
    """Loop interactivo de chat en terminal."""
    console.rule(f"[bold magenta]{title}[/bold magenta]")
    console.print("[dim]Escribe [bold]salir[/bold] para terminar.[/dim]\n")

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]Tú[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in {"salir", "exit", "quit"}:
                break
            await chat(agent, user_input, title=agent_title)
            console.print()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        console.print("\n[dim]¡Hasta luego! 👋[/dim]")



async def run_chat_loop(agent: Agent, title: str = "Agente con Ollama", agent_title: str = "Agente") -> None:
    """Loop interactivo de chat en terminal."""
    console.rule(f"[bold magenta]{title}[/bold magenta]")
    console.print("[dim]Escribe [bold]salir[/bold] para terminar.[/dim]\n")

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]Tú[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in {"salir", "exit", "quit"}:
                break
            await chat(agent, user_input, title=agent_title)
            console.print()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        console.print("\n[dim]¡Hasta luego! 👋[/dim]")


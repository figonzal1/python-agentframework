import asyncio
import logging
import random
from typing import Annotated

from pydantic import Field
from rich.align import Align
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text
from rich.live import Live

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

from ui import console, run_chat_loop

logger = logging.getLogger(__name__)

# ── Client & tools ────────────────────────────────────────────────────────────

client = OpenAIChatClient(
    model_id="qwen2.5:7b",  # ollama pull qwen2.5
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


@tool
def get_weather(
    city: Annotated[str, Field(description="Nombre de la ciudad para la que se desea obtener el clima")],
) -> str:
    """Devuelve datos del clima para una ciudad."""
    logger.info(f"Obteniendo el clima para {city}")
    conditions = ["soleado", "nublado", "lluvioso", "tormentoso"]
    return f"El clima en {city} está {conditions[random.randint(0, 3)]} con una máxima de {random.randint(10, 30)}°C."


agent = Agent(
    client=client,
    instructions="Eres un agente de clima que responde a preguntas sobre el clima actual en diferentes ciudades. Usa las herramientas para obtener esta información. Responde en español con buena onda.",
    tools=[get_weather],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """Print a labelled section divider."""
    console.print()
    console.print(Rule(title=f"[bold #94A3B8]{title}[/]", style="#334155"))
    console.print()


def _human(msg: str) -> None:
    label = Text()
    label.append("  Human  ", style="h.prefix")
    label.append(msg, style="bold white")
    console.print(label)


async def _ask(msg: str, session=None) -> str:
    """Send a message to the agent with a live thinking indicator, return response text."""
    kwargs = {"session": session} if session is not None else {}

    agent_label = Text()
    agent_label.append("  ◆ Agente ", style="a.prefix")
    console.print(agent_label)

    response_text = ""
    spinner = Spinner("dots", text=Text("    thinking…", style="thinking"), style="thinking")

    with Live(spinner, console=console, refresh_per_second=20, transient=True):
        result = await agent.run(msg, **kwargs)
        response_text = result.text or ""

    console.print(Padding(Text(response_text), (0, 0, 0, 4)))
    console.print()
    return response_text


# ── Examples ──────────────────────────────────────────────────────────────────

async def example_without_session() -> None:
    """Sin sesión: cada llamada es independiente, el agente no recuerda mensajes previos."""
    _section("Sin sesión  —  sin memoria")

    q1 = "¿Cómo está el clima en Seattle?"
    _human(q1)
    await _ask(q1)

    q2 = "¿Cuál fue la última ciudad por la que pregunté?"
    _human(q2)
    await _ask(q2)


async def example_with_session() -> None:
    """Con sesión: el agente mantiene contexto a través de varios mensajes."""
    _section("Con sesión  —  memoria persistente")

    session = agent.create_session()

    q1 = "¿Cómo está el clima en Tokio?"
    _human(q1)
    await _ask(q1, session=session)

    q2 = "¿Y en Londres?"
    _human(q2)
    await _ask(q2, session=session)

    q3 = "¿Cuál de esas ciudades tiene mejor clima?"
    _human(q3)
    await _ask(q3, session=session)


async def example_session_across_agents() -> None:
    """Una sesión se puede compartir entre distintas instancias de agente."""
    _section("Sesión compartida entre instancias")

    session = agent.create_session()

    q1 = "¿Cómo está el clima en París?"
    _human(q1)
    await _ask(q1, session=session)

    # Segundo agente con distintas instrucciones, misma sesión
    agent2 = Agent(
        client=client,
        instructions="Eres un agente de clima útil.",
        tools=[get_weather],
    )

    q2 = "¿Cuál fue la última ciudad por la que pregunté?"
    label = Text()
    label.append("  Human  ", style="h.prefix")
    label.append(q2, style="bold white")
    console.print(label)

    agent2_label = Text()
    agent2_label.append("  ◆ Agente 2 ", style="a.prefix")
    console.print(agent2_label)

    spinner = Spinner("dots", text=Text("    thinking…", style="thinking"), style="thinking")
    with Live(spinner, console=console, refresh_per_second=20, transient=True):
        result = await agent2.run(q2, session=session)

    console.print(Padding(Text(result.text or ""), (0, 0, 0, 4)))
    console.print()


# ── Entry points ──────────────────────────────────────────────────────────────

async def main() -> None:
    """Runs both demos back-to-back to show the difference."""
    banner = Text(justify="center")
    banner.append("◆  Sesiones\n", style="banner.title")
    banner.append("Comparativa con y sin memoria de conversación", style="banner.sub")

    console.print()
    console.print(
        Align.center(
            Panel(
                Align.center(banner),
                border_style="banner.border",
                padding=(1, 6),
                expand=False,
            )
        )
    )

    await example_without_session()
    await example_with_session()
    await example_session_across_agents()

    console.print()
    console.print(Align.center(Text("Done", style="hint")))
    console.print()


async def main_interactive() -> None:
    """Interactive chat loop with session memory."""
    await run_chat_loop(agent, title="Agente con Sesión", agent_title="Agente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

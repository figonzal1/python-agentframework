import asyncio
from datetime import datetime
import logging
import random
from typing import Annotated

from pydantic import Field

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

from ui import run_chat_loop

logger = logging.getLogger(__name__)

# llama3.2 (3B) tiene soporte de tools muy limitado.
# Modelos con buen tool calling en Ollama: llama3.1, qwen2.5, mistral
client = OpenAIChatClient(
    model_id="qwen2.5:7b",  # ollama pull qwen2.5
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

@tool
def get_weather(
    city: Annotated[str, Field(description="Nombre de la ciudad para la que se desea obtener el clima")],
) -> dict:
    """Devuelve datos meteorológicos para una ciudad: temperatura y descripción."""
    logger.info(f"Obteniendo el clima para {city}")
    if random.random() < 0.5:
        return {
            "temperature": 72,
            "description": "Soleado",
        }
    else:
        return {
            "temperature": 60,
            "description": "Lluvioso",
        }

@tool
def get_activities(
    city: Annotated[str, Field(description="Ciudad para la que se desean obtener actividades")],
    date: Annotated[str, Field(description="Fecha (YYYY-MM-DD) para la que se desean obtener actividades")],
) -> list[dict]:
    """Devuelve una lista de actividades para una ciudad y fecha dadas."""
    logger.info(f"Obteniendo actividades para {city} en {date}")
    return [
        {"name": "Senderismo", "location": city},
        {"name": "Playa", "location": city},
        {"name": "Museo", "location": city},
    ]


@tool
def get_current_date() -> str:
    """Obtiene la fecha actual del sistema en formato YYYY-MM-DD."""
    logger.info("Obteniendo la fecha actual")
    return datetime.now().strftime("%Y-%m-%d")

agent = Agent(
    client=client,
    instructions="Eres un agente de clima y actividades que responde a preguntas sobre el clima actual y actividades en diferentes ciudades. Usa las herramientas para obtener esta información. Responde en español con buena onda.",
    tools=[get_weather, get_activities, get_current_date],
)

async def main() -> None:
    await run_chat_loop(agent, title="Agente + Multi-Tools")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

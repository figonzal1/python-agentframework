import asyncio
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
    model_id="qwen2.5",  # ollama pull qwen2.5
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

@tool
def get_weather(
    city: Annotated[str, Field(description="City name")],
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


agent = Agent(
    client=client,
    instructions="Eres un agente informativo. Responde a las preguntas con buena onda.",
    tools=[get_weather],
)

async def main() -> None:
    await run_chat_loop(agent, title="Agente con Tools")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

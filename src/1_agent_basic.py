import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from ui import run_chat_loop

client = OpenAIChatClient(
    model_id="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

agent = Agent(
    client=client,
    instructions="Eres un agente informativo. Responde a las preguntas con buena onda.",
)


async def main() -> None:
    await run_chat_loop(agent, title="Agente Básico")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

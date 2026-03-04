import asyncio
import sys

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder

from ui import run_workflow_loop

# Crea los agentes de IA — se pasan directamente como ejecutores al SequentialBuilder,
# igual que las subclases de Executor en workflow_rag_ingest.py.

client = OpenAIChatClient(
    model_id="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

writer = Agent(
    client=client,
    name="Escritor",
    instructions=(
        "Eres un escritor de contenido conciso. "
        "Escribe un artículo corto (2-3 párrafos) claro y atractivo sobre el tema del usuario. "
        "Prioriza la precisión y la legibilidad."
    ),
)

reviewer = Agent(
    client=client,
    name="Revisor",
    instructions=(
        "Eres un revisor de contenido reflexivo. "
        "Lee el borrador del escritor y ofrece retroalimentación específica y constructiva. "
        "Comenta sobre la claridad, la precisión y la estructura. Mantén tu revisión concisa."
    ),
)

workflow = SequentialBuilder(participants=[writer, reviewer]).build()

if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8092, auto_open=True)
    else:
        try:
            asyncio.run(
                run_workflow_loop(
                    workflow,
                    agents=[writer, reviewer],
                    title="Escritor → Revisor (Sequential)",
                )
            )
        except KeyboardInterrupt:
            pass
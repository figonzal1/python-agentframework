import asyncio
import sys

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder
from agent_framework_orchestrations import ConcurrentBuilder

from ui import run_workflow_loop

# Crea los agentes de IA — se pasan directamente como ejecutores al SequentialBuilder,
# igual que las subclases de Executor en workflow_rag_ingest.py.

client = OpenAIChatClient(
    model_id="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Tres agentes especialistas — cada uno aporta una perspectiva diferente al mismo prompt
researcher = Agent(
    client=client,
    name="Investigador",
    instructions=(
        "Eres un experto en investigación de mercado y productos. "
        "Dado un prompt, proporciona información concisa, factual, oportunidades y riesgos. "
        "Limita tu análisis a un párrafo."
    ),
)

marketer = Agent(
    client=client,
    name="Mercadólogo",
    instructions=(
        "Eres un estratega creativo de marketing. "
        "Elabora una propuesta de valor atractiva y mensajes dirigidos alineados con el prompt. "
        "Limita tu respuesta a un párrafo."
    ),
)

legal = Agent(
    client=client,
    name="Legal",
    instructions=(
        "Eres un revisor cauteloso de asuntos legales y cumplimiento normativo. "
        "Destaca restricciones, advertencias y preocupaciones de política basadas en el prompt. "
        "Limita tu respuesta a un párrafo."
    ),
)

# Construye el workflow concurrente — los tres agentes se ejecutan en paralelo
workflow = ConcurrentBuilder(participants=[researcher, marketer, legal]).build()

if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8092, auto_open=True)
    else:
        try:
            asyncio.run(
                run_workflow_loop(
                    workflow,
                    agents=[researcher, marketer, legal],
                    title="Investigador → Mercadólogo → Legal (Concurrente)",
                )
            )
        except KeyboardInterrupt:
            pass
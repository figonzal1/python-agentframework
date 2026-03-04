import asyncio
import sys
from typing import Any, Never

from agent_framework import Agent, AgentExecutorResponse, WorkflowBuilder, WorkflowContext, executor
from agent_framework.openai import OpenAIChatClient
from typing import Any, Literal
from pydantic import BaseModel

from ui import run_workflow_loop

class ReviewDecision(BaseModel):
    """Decisión estructurada del revisor para enrutamiento condicional."""

    decision: Literal["APPROVED", "REVISION_NEEDED"]
    feedback: str
    post_text: str | None = None

# Helper de parseo para mantener pequeñas y explícitas las funciones de condición.
def parse_review_decision(message: Any) -> ReviewDecision | None:
    """Parsea la salida estructurada del revisor desde AgentExecutorResponse."""
    if not isinstance(message, AgentExecutorResponse):
        return None

    return ReviewDecision.model_validate_json(message.agent_response.text)


def is_approved(message: Any) -> bool:
    """Enruta al publicador cuando la decisión estructurada es APPROVED."""
    result = parse_review_decision(message)
    return result is not None and result.decision == "APPROVED"


def needs_revision(message: Any) -> bool:
    """Enruta al editor cuando la decisión estructurada es REVISION_NEEDED."""
    result = parse_review_decision(message)
    return result is not None and result.decision == "REVISION_NEEDED"

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
        "Escribe un artículo corto (2-3 párrafos) claro y atractivo sobre el tema del usuario."
    ),
)

reviewer = Agent(
    client=client,
    name="Revisor",
    instructions=(
        "Eres un revisor de contenido estricto. Evalúa el borrador del escritor. "
        "Si el borrador está listo, define decision=APPROVED e incluye la publicación lista para publicar en post_text. "
        "Si necesita cambios, define decision=REVISION_NEEDED y entrega feedback accionable."
    ),
    default_options={"response_format": ReviewDecision},
)

editor = Agent(
    client=client,
    name="Editor",
    instructions=(
        "Eres un editor habilidoso. "
        "Recibes un borrador del escritor seguido de la retroalimentación del revisor. "
        "Reescribe el borrador abordando todos los problemas señalados. "
        "Entrega solo el artículo mejorado."
    ),
)

@executor(id="publisher")
async def publisher(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """Publica contenido desde la salida estructurada del revisor."""
    result = parse_review_decision(response)
    if result is None:
        await ctx.yield_output("✅ Publicado:\n\n(No se pudo parsear la salida estructurada del revisor.)")
        return

    content = (result.post_text or "").strip()
    if not content:
        content = "(El revisor aprobó pero no incluyó post_text.)"

    await ctx.yield_output(f"✅ Publicado:\n\n{content}")


workflow = (
    WorkflowBuilder(start_executor=writer)
    .add_edge(writer, reviewer)
    .add_edge(reviewer, publisher, condition=is_approved)
    .add_edge(reviewer, editor, condition=needs_revision)
    .build()
)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8092, auto_open=True)
    else:
        try:
            asyncio.run(
                run_workflow_loop(
                    workflow,
                    agents=[writer, reviewer, editor],
                    title="Escritor → Revisor → aprobado: Publicar / revisar: Editor",
                )
            )
        except KeyboardInterrupt:
            pass
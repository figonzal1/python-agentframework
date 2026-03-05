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
        "Eres un revisor de contenido estricto. Evalúa el borrador del escritor. El contenido debe estar en español. "
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

# Ejecutor de estado tipo pass-through: guarda el último texto de publicación de Escritor/Editor.
# Docs: https://learn.microsoft.com/en-us/agent-framework/workflows/state?pivots=programming-language-python
@executor(id="store_post_text")
async def store_post_text(response: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorResponse]) -> None:
    """Guarda el último texto de publicación en el estado del workflow y lo pasa aguas abajo."""
    ctx.set_state("post_text", response.agent_response.text.strip())
    await ctx.send_message(response)


# Ejecutor terminal: recibe la decisión estructurada del revisor y publica desde estado.
# Se usa @executor para una función independiente en lugar de una subclase de Executor —
# ambas son formas válidas de definir un nodo en el workflow.
@executor(id="publisher")
async def publisher(_response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """Publica el último texto de publicación aprobado desde el estado del workflow."""
    content = str(ctx.get_state("post_text", "")).strip()
    await ctx.yield_output(f"✅ Publicado:\n\n{content}")


# Construye el workflow con dos aristas condicionales de salida desde el revisor.
# add_edge(reviewer, publisher, condition=is_approved) se activa cuando is_approved() retorna True.
# add_edge(reviewer, editor, condition=needs_revision) se activa cuando needs_revision() retorna True.
# add_edge(editor, reviewer) crea un bucle acotado de revisión-edición.
workflow = (
    WorkflowBuilder(start_executor=writer, max_iterations=20)
    .add_edge(writer, store_post_text)
    .add_edge(store_post_text, reviewer)
    .add_edge(reviewer, publisher, condition=is_approved)
    .add_edge(reviewer, editor, condition=needs_revision)
    .add_edge(editor, store_post_text)
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
                    title="Escritor → Revisor → aprobado: Publicar / revisar: Editor ↻",
                    workflow_title="Escritor → Revisor (con estado y bucle de revisión)",
                )
            )
        except KeyboardInterrupt:
            pass
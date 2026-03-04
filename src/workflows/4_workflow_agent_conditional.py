import asyncio
import sys
from typing import Any, Never

from agent_framework import Agent, AgentExecutorResponse, WorkflowBuilder, WorkflowContext, executor
from agent_framework.openai import OpenAIChatClient

from ui import run_workflow_loop

# Crea los agentes de IA — se pasan directamente como ejecutores al SequentialBuilder,
# igual que las subclases de Executor en workflow_rag_ingest.py.

client = OpenAIChatClient(
    model_id="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Funciones de condición — reciben el mensaje del ejecutor anterior.
# Ambas verifican con isinstance() ya que las condiciones pueden recibir cualquier tipo.
def is_approved(message: Any) -> bool:
    """Enruta al publicador si el revisor comenzó su respuesta con APPROVED."""
    if not isinstance(message, AgentExecutorResponse):
        return False
    return message.agent_response.text.upper().startswith("APPROVED")


def needs_revision(message: Any) -> bool:
    """Enruta al editor si el revisor solicitó cambios."""
    if not isinstance(message, AgentExecutorResponse):
        return False
    return message.agent_response.text.upper().startswith("REVISION NEEDED")


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
        "Eres un revisor de contenido estricto. Evalúa el borrador del escritor.\n"
        "Verifica que la publicación sea atractiva y adecuada para la plataforma objetivo.\n"
        "Asegúrate de que no suene demasiado generada por LLM.\n"
        "Restricciones de estilo/accesibilidad: no uses em dash (—) y no uses texto Unicode sofisticado.\n"
        "IMPORTANTE: Tu respuesta DEBE comenzar con exactamente uno de estos dos tokens:\n"
        "  APPROVED        — si el borrador es claro, preciso y bien estructurado.\n"
        "  REVISION NEEDED — si necesita mejoras.\n"
        "Si eliges APPROVED, incluye la publicación final inmediatamente después del token.\n"
        "Si eliges REVISION NEEDED, proporciona una breve explicación de qué corregir."
    ),
)

editor = Agent(
    client=client,
    name="Editor",
    instructions=(
        "Eres un editor habilidoso. "
        "Recibes un borrador del escritor seguido de la retroalimentación del revisor. "
        "Reescribe el borrador abordando todos los problemas señalados. "
        "Entrega solo el artículo mejorado."
        "Asegúrate de que el largo de la publicación final sea apropiado para la plataforma objetivo."
    ),
)

# Ejecutor terminal: recibe la respuesta APPROVED del revisor y la publica.
# Se usa @executor para una función independiente en lugar de una subclase de Executor —
# ambas son formas válidas de definir un nodo en el workflow.
@executor(id="publisher")
async def publisher(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """Elimina el prefijo APPROVED y entrega el contenido publicado final."""
    text = response.agent_response.text
    content = text[len("APPROVED") :].lstrip(":").strip()

    await ctx.yield_output(f"✅ Publicado:\n\n{content}")


# Construye el workflow con dos aristas condicionales de salida desde el revisor.
# add_edge(reviewer, publisher, condition=is_approved) se activa cuando is_approved() retorna True.
# add_edge(reviewer, editor, condition=needs_revision) se activa cuando needs_revision() retorna True.
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
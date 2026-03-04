"""Workflow terminal UI — polished Rich CLI for multi-agent workflows."""
import asyncio
import time
from datetime import datetime
from typing import Any, Sequence

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text
from rich.theme import Theme

from agent_framework import Agent

# ── Theme ─────────────────────────────────────────────────────────────────────
_THEME = Theme({
    "h.prefix":      "bold #7C9EFF",
    "a.prefix":      "bold #FFFFFF",
    "step.icon":     "#7C9EFF",
    "step.name":     "bold #7C9EFF",
    "step.index":    "dim #7C9EFF",
    "workflow.icon": "#A78BFA",
    "workflow.name": "bold #A78BFA",
    "spinner":       "#A78BFA",
    "tool.done":     "#52C41A",
    "thinking":      "dim #6B7280",
    "ts":            "dim #4B5563",
    "hint":          "dim #6B7280",
    "divider":       "#1F2937",
    "meta":          "dim #374151",
    "interrupted":   "dim #EF4444",
    "banner.title":  "bold #E2E8F0",
    "banner.sub":    "dim #94A3B8",
    "banner.border": "#334155",
    "step.border":   "#4B5563",
})

console = Console(theme=_THEME, highlight=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _human_label() -> Text:
    t = Text()
    t.append("  Human ", style="h.prefix")
    t.append(_ts(), style="ts")
    return t


def _agent_label(name: str, index: int, total: int) -> Text:
    t = Text()
    t.append(f"  ◆ {name} ", style="a.prefix")
    t.append(f"[{index}/{total}] ", style="step.index")
    t.append(_ts(), style="ts")
    return t


def _extract_agent_turns(output: Any, agents: Sequence["Agent"]) -> list[tuple[str, str]]:
    """Return [(agent_name, text), ...] for every assistant turn in output."""
    # AgentExecutorResponse — from WorkflowBuilder nodes
    if hasattr(output, "agent_response") and hasattr(output, "executor_id"):
        text = output.agent_response.text or ""
        # Match executor_id to an agent name; fall back to executor_id itself
        name = next(
            (a.name for a in agents if (a.name or "").lower() == output.executor_id.lower()),
            output.executor_id,
        )
        return [(name, text)]
    # List of Message objects (full conversation) — from SequentialBuilder
    if isinstance(output, list) and output and hasattr(output[0], "role"):
        assistant_texts = [
            m.text for m in output
            if hasattr(m, "role") and m.role == "assistant" and hasattr(m, "text") and m.text
        ]
        result = []
        for i, text in enumerate(assistant_texts):
            name = agents[i].name if i < len(agents) and agents[i].name else f"Agente {i + 1}"
            result.append((name, text))
        return result
    # Single Message object
    if hasattr(output, "role") and hasattr(output, "text"):
        name = agents[0].name if agents and agents[0].name else "Agente"
        return [(name, output.text or str(output))]
    # Plain string — output from ctx.yield_output (e.g. publisher executor)
    if isinstance(output, str):
        return [("Publicado", output)]
    # Fallback
    return [("Publicado", str(output))]


def _workflow_spinner(name: str) -> Spinner:
    label = Text()
    label.append("  ◆ ", style="workflow.icon")
    label.append(name, style="workflow.name")
    label.append("  running…", style="thinking")
    return Spinner("dots2", text=label, style="spinner")


# ── Core workflow runner ───────────────────────────────────────────────────────

async def run_workflow(
    workflow: Any,
    prompt: str,
    agents: Sequence[Agent],
    workflow_title: str = "Workflow",
) -> None:
    """Runs the workflow, shows a spinner, then prints each agent's output sequentially."""
    t0 = time.monotonic()

    # ── Spinner while workflow executes ──────────────────────────────────────
    result = None
    with Live(
        _workflow_spinner(workflow_title),
        console=console,
        refresh_per_second=20,
        transient=True,
    ):
        result = await workflow.run(prompt)

    # ── Print each agent's output ─────────────────────────────────────────────
    outputs = result.get_outputs() if result else []
    total_agents = len(agents)

    # Flatten all outputs into (agent_name, text) turns
    turns: list[tuple[str, str]] = []
    for output in outputs:
        turns.extend(_extract_agent_turns(output, agents[len(turns):]))

    total = len(turns)
    for i, (name, text) in enumerate(turns):
        elapsed = time.monotonic() - t0

        console.print(_agent_label(name, i + 1, total))
        console.print(Padding(Markdown(text), (0, 0, 0, 2)))

        wc = len(text.split())
        meta = Text()
        meta.append(f"    {wc} words  ·  {elapsed:.1f}s elapsed", style="meta")
        console.print(meta)
        console.print(Rule(style="divider"))
        console.print()


# ── Interactive workflow loop ─────────────────────────────────────────────────

async def run_workflow_loop(
    workflow: Any,
    agents: Sequence[Agent],
    title: str = "Workflow",
    workflow_title: str | None = None,
) -> None:
    """Interactive loop: accepts a user prompt and runs the full workflow pipeline."""
    wf_title = workflow_title or title

    # ── Banner ────────────────────────────────────────────────────────────────
    agent_names = " → ".join(
        a.name or f"Agente {i + 1}" for i, a in enumerate(agents)
    )
    banner_text = Text(justify="center")
    banner_text.append(f"◆  {title}\n", style="banner.title")
    banner_text.append(agent_names + "\n", style="banner.sub")
    banner_text.append("ctrl+c  ·  exit to quit", style="banner.sub")

    console.print()
    console.print(
        Align.center(
            Panel(
                Align.center(banner_text),
                border_style="banner.border",
                padding=(1, 6),
                expand=False,
            )
        )
    )
    console.print()

    try:
        while True:
            console.print(_human_label())
            try:
                user_input = Prompt.ask(
                    Text("  >", style="h.prefix"),
                    console=console,
                )
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue
            if user_input.strip().lower() in {"salir", "exit", "quit", "q"}:
                break

            console.print()
            try:
                await run_workflow(workflow, user_input, agents, workflow_title=wf_title)
            except (KeyboardInterrupt, asyncio.CancelledError):
                console.print(Text("    interrupted", style="interrupted"))
                console.print(Rule(style="divider"))
                console.print()

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        console.print()
        console.print(Align.center(Text("Session ended", style="hint")))
        console.print()

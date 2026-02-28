"""Shared terminal UI for agent demos — polished Claude-inspired CLI."""
import asyncio
import json
import time
from datetime import datetime

from rich.align import Align
from rich.console import Console, Group
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

# ── Global theme ──────────────────────────────────────────────────────────────
_THEME = Theme({
    "h.prefix":      "bold #7C9EFF",   # human prefix  — Claude blue
    "a.prefix":      "bold #FFFFFF",   # agent prefix  — white
    "tool.icon":     "#F5A623",        # tool icon     — amber
    "tool.name":     "italic #F5A623", # tool name     — amber italic
    "tool.args":     "dim #C0834F",    # tool args     — muted amber
    "tool.done":     "#52C41A",        # tool ✓        — green
    "thinking":      "dim #6B7280",    # thinking      — grey
    "ts":            "dim #4B5563",    # timestamps    — dark grey
    "hint":          "dim #6B7280",    # hints         — grey
    "divider":       "#1F2937",        # dividers      — very dark
    "meta":          "dim #374151",    # word count    — darkest
    "interrupted":   "dim #EF4444",   # interrupt     — red
    "banner.title":  "bold #E2E8F0",  # banner title  — near white
    "banner.sub":    "dim #94A3B8",   # banner sub    — slate
    "banner.border": "#334155",       # banner border — slate dark
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


def _agent_label(title: str) -> Text:
    t = Text()
    t.append(f"  ◆ {title} ", style="a.prefix")
    t.append(_ts(), style="ts")
    return t


def _parse_args(arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            pass
    if isinstance(arguments, dict):
        return ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in arguments.items()
        )
    return str(arguments) if arguments else ""


def _build_tool_spinner(name: str, args_str: str) -> Spinner:
    """Animated spinner shown while a tool is running."""
    label = Text()
    label.append("    ⚡ ", style="tool.icon")
    label.append(name, style="tool.name")
    label.append(f"({args_str})", style="tool.args")
    return Spinner("dots2", text=label, style="tool.icon")


def _build_tool_done(name: str, args_str: str) -> Text:
    """Static check line once a tool call completes."""
    t = Text()
    t.append("    ✓ ", style="tool.done")
    t.append(name, style="tool.name")
    t.append(f"({args_str})", style="tool.args")
    return t


def _thinking_renderable() -> Spinner:
    return Spinner("dots", text=Text("    thinking…", style="thinking"), style="thinking")


# ── Core chat function ────────────────────────────────────────────────────────

async def chat(agent: Agent, user_input: str, title: str = "Agente") -> None:
    """Streams the agent response with animated tool calls and progressive markdown."""
    collected = ""
    t0 = time.monotonic()

    # call_id → (name, args_str)
    tool_calls: dict[str, tuple[str, str]] = {}
    tool_order: list[str] = []

    def _live_renderable():
        parts: list = []
        if not collected:
            for cid in tool_order:
                name, args_str = tool_calls[cid]
                parts.append(_build_tool_spinner(name, args_str))
            if not tool_order:
                parts.append(_thinking_renderable())
        else:
            for cid in tool_order:
                name, args_str = tool_calls[cid]
                parts.append(_build_tool_done(name, args_str))
            parts.append(Markdown(collected))
        return Group(*parts) if len(parts) > 1 else (parts[0] if parts else _thinking_renderable())

    stream = agent.run(user_input, stream=True)
    console.print(_agent_label(title))

    try:
        with Live(
            _live_renderable(),
            console=console,
            refresh_per_second=20,
            transient=True,
            vertical_overflow="visible",
        ) as live:
            async for update in stream:
                for content in update.contents or []:
                    if content.type == "function_call" and content.call_id not in tool_calls:
                        args_str = _parse_args(content.arguments)
                        tool_calls[content.call_id] = (content.name, args_str)
                        tool_order.append(content.call_id)
                        live.update(_live_renderable())

                if update.text:
                    collected += update.text
                    live.update(_live_renderable())

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print(Text("    interrupted", style="interrupted"))

    # ── Permanent output (after Live exits) ──────────────────────────────────
    for cid in tool_order:
        name, args_str = tool_calls[cid]
        console.print(_build_tool_done(name, args_str))

    if collected:
        console.print(Padding(Markdown(collected), (0, 0, 0, 2)))
        elapsed = time.monotonic() - t0
        wc = len(collected.split())
        meta = Text()
        meta.append(f"    {wc} words  ·  {elapsed:.1f}s", style="meta")
        console.print(meta)

    console.print(Rule(style="divider"))
    console.print()


# ── Chat loop ─────────────────────────────────────────────────────────────────

async def run_chat_loop(
    agent: Agent,
    title: str = "Agente con Ollama",
    agent_title: str = "Agente",
) -> None:
    """Interactive chat loop with styled banner and turn management."""

    # ── Startup banner ───────────────────────────────────────────────────────
    banner_text = Text(justify="center")
    banner_text.append(f"◆  {title}\n", style="banner.title")
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
            await chat(agent, user_input, title=agent_title)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        console.print()
        console.print(Align.center(Text("Session ended", style="hint")))
        console.print()


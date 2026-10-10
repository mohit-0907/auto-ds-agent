"""Regenerate the README's graph diagram from the real graph.

    uv run python scripts/render_graph.py

Writes docs/graph.mmd (the Mermaid source, checked against the code by
tests/test_graph.py) and docs/graph.png (the image the README shows).

The PNG is rendered by LangGraph's draw_mermaid_png(), which sends the Mermaid
text to the public mermaid.ink service -- so this needs network access, and
only the graph's structure (node names and edges) leaves the machine.
"""

from pathlib import Path

import structlog

from autods.graph import build_graph

DOCS = Path(__file__).resolve().parent.parent / "docs"

log = structlog.get_logger(__name__)


def main() -> None:
    """Write docs/graph.mmd and docs/graph.png."""
    drawable = build_graph().get_graph()
    DOCS.mkdir(exist_ok=True)
    (DOCS / "graph.mmd").write_text(drawable.draw_mermaid(), encoding="utf-8")
    (DOCS / "graph.png").write_bytes(drawable.draw_mermaid_png())
    log.info("rendered", mmd=str(DOCS / "graph.mmd"), png=str(DOCS / "graph.png"))


if __name__ == "__main__":
    main()

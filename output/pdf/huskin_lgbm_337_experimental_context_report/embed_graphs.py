"""Embed the source-backed analysis graphs in the portable HuSkinDB report."""

from __future__ import annotations

import base64
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
SOURCE_HTML = REPORT_DIR / "report.html"
OUTPUT_HTML = REPORT_DIR / "report_with_graphs.html"
GRAPH_DIR = (
    PROJECT_ROOT
    / "results"
    / "huskinDB"
    / "within_0.6_LGBM"
    / "experimental_context"
    / "analysis_337"
)

GRAPHS = {
    "compound_variability_chart": (
        GRAPH_DIR / "compound_variability.png",
        "Horizontal bar chart of the largest within-compound experimental logKp ranges.",
    ),
    "condition_missingness_chart": (
        GRAPH_DIR / "condition_missingness.png",
        "Horizontal bar chart of unavailable experimental-condition fields in the selected rows.",
    ),
    "reference_coverage_chart": (
        GRAPH_DIR / "reference_coverage.png",
        "Horizontal bar chart of the references contributing the most selected measurement rows.",
    ),
    "donor_ph_chart": (
        GRAPH_DIR / "donor_ph_series.png",
        "Line chart of experimental logKp across reported donor-pH series for three compounds.",
    ),
}

EMBEDDED_GRAPH_CSS = """
/* Source-backed figures embedded for static report and PDF delivery. */
.embedded-static-graph {
  margin: 10px 0 8px;
  text-align: center;
}
.embedded-static-graph img {
  display: block;
  width: 100%;
  height: auto;
  margin: 0 auto;
  border: 1px solid var(--portable-border);
  border-radius: 8px;
  background: #fff;
}
figure[data-static-graph-embedded="true"] > .portable-table-scroll {
  margin-top: 10px;
}
@media print {
  .embedded-static-graph {
    margin: 6px 0 4px;
    break-inside: avoid;
  }
  .embedded-static-graph img {
    max-height: 160mm;
    object-fit: contain;
  }
  figure[data-static-graph-embedded="true"] {
    break-inside: auto !important;
  }
}
"""


def image_data_uri(path: Path) -> str:
    """Return a PNG as a self-contained data URI."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def embed_graph(html: str, chart_id: str, path: Path, alt_text: str) -> str:
    """Insert one graph after its figure caption while preserving fallback data."""

    if not path.is_file():
        raise FileNotFoundError(f"Graph not found: {path}")

    pattern = re.compile(
        rf'(<figure\b(?=[^>]*data-chart-id="{re.escape(chart_id)}")[^>]*>)(.*?)(</figure>)',
        flags=re.DOTALL,
    )
    match = pattern.search(html)
    if match is None:
        raise ValueError(f"Chart figure not found in report: {chart_id}")

    opening, body, closing = match.groups()
    opening = opening[:-1] + ' data-static-graph-embedded="true">'
    graph_markup = (
        '<div class="embedded-static-graph">'
        f'<img src="{image_data_uri(path)}" alt="{alt_text}" loading="eager">'
        "</div>"
    )
    if "</figcaption>" not in body:
        raise ValueError(f"Figure caption not found for chart: {chart_id}")
    body = body.replace("</figcaption>", f"</figcaption>{graph_markup}", 1)
    return html[: match.start()] + opening + body + closing + html[match.end() :]


def main() -> None:
    """Build the self-contained report HTML containing all four graphs."""

    html = SOURCE_HTML.read_text(encoding="utf-8")
    if "</style>" not in html:
        raise ValueError("The report does not contain a style block.")
    html = html.replace("</style>", f"{EMBEDDED_GRAPH_CSS}\n</style>", 1)

    for chart_id, (graph_path, alt_text) in GRAPHS.items():
        html = embed_graph(html, chart_id, graph_path, alt_text)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()

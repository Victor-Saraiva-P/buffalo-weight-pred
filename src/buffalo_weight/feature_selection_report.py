"""Auditable draft for manual shared-feature selection."""

from __future__ import annotations

from buffalo_weight.feature_recommendations import RemovalRecommendation


def selection_report_markdown(recommendations: list[RemovalRecommendation]) -> str:
    """Render every withdrawal result; for example, neutral and rejected rows remain visible."""
    lines = [
        "# Minuta de Seleção do Conjunto Compartilhado de Features",
        "",
        "Este documento organiza evidências OOF pós-seleção para revisão humana. "
        "Nenhuma recomendação abaixo constitui uma decisão automática.",
        "",
        "## Testes de retirada",
        "",
        "| Feature ou grupo | Δ MAE RF (kg) | Δ MAE rede densa (kg) | Recomendação provisória |",
        "|---|---:|---:|---|",
    ]
    lines.extend(_recommendation_line(item) for item in recommendations)
    lines.extend(_review_section())
    return "\n".join(lines) + "\n"


def _recommendation_line(item: RemovalRecommendation) -> str:
    return (f"| `{item.target}` | {item.random_forest_delta_mae_kg:.2f} | "
            f"{item.dense_delta_mae_kg:.2f} | `{item.recommendation}` |")


def _review_section() -> list[str]:
    return [
        "", "## Limitações", "",
        "As mesmas máscaras orientam esta seleção; os valores são evidência de desenvolvimento, "
        "não validação independente em animais novos.",
        "", "## Registro de revisão humana", "",
        "- Status: pendente",
        "- Interpretações aceitas, corrigidas ou rejeitadas: não preenchidas",
        "- Conjunto Compartilhado de Features confirmado: não preenchido",
    ]

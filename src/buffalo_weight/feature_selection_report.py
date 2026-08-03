"""Auditable draft for manual shared-feature selection."""

from __future__ import annotations

from buffalo_weight.feature_evaluation import FeatureEvidence
from buffalo_weight.feature_recommendations import RemovalRecommendation
from buffalo_weight.feature_redundancy import FeatureRedundancy


def selection_report_markdown(
    recommendations: list[RemovalRecommendation], redundancy: list[FeatureRedundancy],
    evidence: list[FeatureEvidence],
) -> str:
    """Render all evidence classes; for example, detailed fold rows remain linked in CSV."""
    lines = _introduction(evidence, redundancy)
    lines.extend(_isolated_section(evidence))
    lines.extend(_redundancy_section(redundancy))
    lines.extend(_permutation_section(evidence))
    lines.extend(_removal_section(recommendations))
    lines.extend(_review_section())
    return "\n".join(lines) + "\n"


def _introduction(
    evidence: list[FeatureEvidence], redundancy: list[FeatureRedundancy]
) -> list[str]:
    return [
        "# Minuta de Seleção do Conjunto Compartilhado de Features", "",
        "Este documento organiza evidências OOF pós-seleção para revisão humana. "
        "Nenhuma recomendação abaixo constitui uma decisão automática.", "",
        f"A tabela detalhada `feature_predictive_evidence.csv` contém {len(evidence)} resultados; "
        f"`feature_redundancy.csv` contém os {len(redundancy)} pares. Os resultados por fold e "
        "repetição permanecem nesses artefatos auditáveis.",
    ]


def _isolated_section(evidence: list[FeatureEvidence]) -> list[str]:
    indexed = _oof_metric_index(evidence, "isolated")
    targets = sorted({target for target, _ in indexed})
    lines = ["", "## Desempenho isolado", "",
             "| Feature | MAE RF (kg) | MAE rede densa (kg) |", "|---|---:|---:|"]
    lines.extend(f"| `{target}` | {indexed[(target, 'random_forest')]:.2f} | "
                 f"{indexed[(target, 'dense')]:.2f} |" for target in targets)
    return lines


def _redundancy_section(redundancy: list[FeatureRedundancy]) -> list[str]:
    structural = [row for row in redundancy if row.structural_relation != "none"]
    strongest = sorted(redundancy, key=_observed_strength, reverse=True)[:10]
    lines = ["", "## Redundância estrutural e observada", "",
             f"Há {len(structural)} pares com relação estrutural declarada. O mapa completo está "
             "em `redundancy_heatmap.png`; os 325 pares permanecem no CSV.", "",
             "| Par | Relação estrutural | Pearson | Spearman |", "|---|---|---:|---:|"]
    lines.extend(_redundancy_line(row) for row in strongest)
    return lines


def _permutation_section(evidence: list[FeatureEvidence]) -> list[str]:
    indexed = _oof_permutation_means(evidence)
    targets = sorted({target for target, _ in indexed})
    lines = ["", "## Efeitos de permutação", "",
             "Cada média resume dez permutações determinísticas OOF; repetições e folds estão "
             "em `feature_predictive_evidence.csv` e `permutation_effects.png`.", "",
             "| Feature | Δ MAE RF (kg) | Δ MAE rede densa (kg) |", "|---|---:|---:|"]
    lines.extend(f"| `{target}` | {indexed[(target, 'random_forest')]:.2f} | "
                 f"{indexed[(target, 'dense')]:.2f} |" for target in targets)
    return lines


def _removal_section(recommendations: list[RemovalRecommendation]) -> list[str]:
    lines = ["", "## Testes de retirada", "",
             "O mapa completo está em `removal_heatmap.png`.", "",
             "| Feature ou grupo | Δ MAE RF (kg) | Δ MAE rede densa (kg) | Recomendação provisória |",
             "|---|---:|---:|---|"]
    lines.extend(_recommendation_line(item) for item in recommendations)
    return lines


def _oof_metric_index(
    evidence: list[FeatureEvidence], experiment: str
) -> dict[tuple[str, str], float]:
    rows = [row for row in evidence if row.scope == "oof" and row.experiment == experiment]
    indexed = {(row.target, row.baseline): row.result_mae_kg for row in rows}
    return indexed


def _oof_permutation_means(
    evidence: list[FeatureEvidence],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    rows = [row for row in evidence if row.scope == "oof" and row.experiment == "permutation"]
    for row in rows:
        if row.delta_mae_kg is not None:
            grouped.setdefault((row.target, row.baseline), []).append(row.delta_mae_kg)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _observed_strength(row: FeatureRedundancy) -> float:
    correlations = [abs(value) for value in (row.pearson, row.spearman) if value is not None]
    strength = max(correlations, default=-1.0)
    return strength


def _redundancy_line(row: FeatureRedundancy) -> str:
    pearson = "" if row.pearson is None else f"{row.pearson:.3f}"
    spearman = "" if row.spearman is None else f"{row.spearman:.3f}"
    return (f"| `{row.feature_a}` / `{row.feature_b}` | `{row.structural_relation}` | "
            f"{pearson} | {spearman} |")


def _recommendation_line(item: RemovalRecommendation) -> str:
    line = (f"| `{item.target}` | {item.random_forest_delta_mae_kg:.2f} | "
            f"{item.dense_delta_mae_kg:.2f} | `{item.recommendation}` |")
    return line


def _review_section() -> list[str]:
    lines = ["", "## Limitações", "",
             "As mesmas máscaras orientam esta seleção; os valores são evidência de desenvolvimento, "
             "não validação independente em animais novos.", "", "## Registro de revisão humana", "",
             "- Status: pendente", "- Interpretações aceitas, corrigidas ou rejeitadas: não preenchidas",
             "- Conjunto Compartilhado de Features confirmado: não preenchido"]
    return lines

"""Draft report for human selection of the most promising approach."""

from __future__ import annotations

from buffalo_weight.baseline_comparison_plots import APPROACH_LABELS, APPROACH_ORDER
from buffalo_weight.baseline_comparison_types import ComparisonMetric, EvaluationRole


def approach_selection_report(metrics: list[ComparisonMetric]) -> str:
    """Render a reviewable draft; for example, the human decision stays unfilled."""
    candidates = _global_rows(metrics, "candidate")
    reference = _global_rows(metrics, "reference")
    recommended = min(candidates, key=lambda row: row.mae_kg)
    lines = _introduction()
    lines.extend(_candidate_table(candidates))
    lines.extend(_extreme_table(metrics))
    lines.extend(_reference_section(reference[0]))
    lines.extend(_recommendation_section(recommended))
    lines.extend(_review_section())
    return "\n".join(lines) + "\n"


def _global_rows(
    metrics: list[ComparisonMetric], role: EvaluationRole,
) -> list[ComparisonMetric]:
    indexed = {row.approach: row for row in metrics if row.evaluation_role == role
               and row.scope == "oof" and row.population == "all"}
    order = APPROACH_ORDER if role == "candidate" else ("training_mean",)
    return [indexed[name] for name in order]


def _introduction() -> list[str]:
    return [
        "# Minuta de Seleção da Abordagem de Maior Potencial", "",
        "Esta minuta organiza o MAE OOF Pós-Seleção e evidências descritivas para revisão "
        "humana. A recomendação abaixo não constitui uma decisão automática.", "",
        "As métricas globais são calculadas diretamente sobre as 132 Predições OOF reunidas; "
        "não são médias simples dos folds.",
    ]


def _candidate_table(rows: list[ComparisonMetric]) -> list[str]:
    lines = ["", "## Quatro abordagens candidatas", "",
             "| Abordagem | MAE (kg) | RMSE (kg) | Viés (kg) | R² |",
             "|---|---:|---:|---:|---:|"]
    lines.extend(
        f"| {APPROACH_LABELS[row.approach]} | {row.mae_kg:.2f} | "
        f"{_number(row.rmse_kg, 2)} | {row.bias_kg:.2f} | {_number(row.r2, 3)} |"
        for row in rows
    )
    return lines


def _extreme_table(metrics: list[ComparisonMetric]) -> list[str]:
    rows = [row for row in metrics if row.evaluation_role == "candidate"
            and row.scope == "oof" and row.population in {"B1", "B10"}]
    indexed = {(row.approach, row.population): row for row in rows}
    lines = ["", "## Evidência descritiva nos extremos", "",
             "| Abordagem | B1 MAE (kg) | B1 viés (kg) | B10 MAE (kg) | B10 viés (kg) |",
             "|---|---:|---:|---:|---:|"]
    for approach in APPROACH_ORDER:
        b1, b10 = indexed[(approach, "B1")], indexed[(approach, "B10")]
        lines.append(f"| {APPROACH_LABELS[approach]} | {b1.mae_kg:.2f} | {b1.bias_kg:.2f} | "
                     f"{b10.mae_kg:.2f} | {b10.bias_kg:.2f} |")
    return lines


def _reference_section(reference: ComparisonMetric) -> list[str]:
    return ["", "## Referência", "",
            f"O preditor da média do treino de cada fold obteve MAE de {reference.mae_kg:.2f} kg. "
            "Ele é uma referência trivial e não uma quinta candidata."]


def _recommendation_section(recommended: ComparisonMetric) -> list[str]:
    label = APPROACH_LABELS[recommended.approach]
    return ["", "## Recomendação revisável", "",
            f"Pelo critério principal predefinido, `{label}` apresenta o menor MAE OOF "
            f"Pós-Seleção ({recommended.mae_kg:.2f} kg) e deve ser priorizada na revisão. "
            "RMSE, viés, R² e os extremos permanecem evidências descritivas; não há pontuação "
            "combinada, bootstrap ou custo computacional como critério."]


def _review_section() -> list[str]:
    return ["", "## Limitações", "",
            "As mesmas 132 máscaras orientaram seleção de features e comparação. Estes valores "
            "são evidência de desenvolvimento, não validação independente em animais novos. "
            "B10 está confundida com fazenda e aquisição na amostra atual.", "",
            "## Registro de revisão humana", "", "- Status: pendente",
            "- Interpretações aceitas, corrigidas ou rejeitadas: não preenchidas",
            "- Decisão humana: não preenchida"]


def _number(value: float | None, decimals: int) -> str:
    if value is None:
        return ""
    formatted = f"{value:.{decimals}f}"
    return formatted

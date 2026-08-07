"""Output artifact writing for controlled sensitivity diagnostic slice.

Reference: GitHub Issue #26.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from buffalo_weight.csv_io import format_csv_number, write_csv_rows
from buffalo_weight.diagnostic_sensitivity_types import (
    MorphologyEligibility,
    SensitivityPerturbationRecord,
    SensitivitySlice,
)


SENSITIVITY_COLUMNS = [
    "configuration",
    "evaluation_scope",
    "file_name",
    "perturbation",
    "status",
    "rejection_reason",
    "original_prediction_kg",
    "perturbed_prediction_kg",
    "delta_kg",
]

ELIGIBILITY_COLUMNS = [
    "file_name",
    "status",
    "rejection_reason",
]


def write_sensitivity_artifacts(
    output_dir: Path,
    slice_data: SensitivitySlice,
    masks_for_demo: dict[str, np.ndarray] | None = None,
) -> None:
    """Write tidy CSV files, demo PNG, and markdown report for sensitivity diagnostics.

    Example: ``write_sensitivity_artifacts(output_dir, slice_data)`` saves all artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sensitivity_csv(output_dir / "sensitivity_perturbations.csv", slice_data)
    _write_eligibility_csv(output_dir / "morphology_eligibility.csv", slice_data)
    if masks_for_demo:
        _write_demo_png(output_dir / "morphology_demo.png", masks_for_demo, slice_data)
    _write_report_markdown(output_dir / "sensitivity_report.md", slice_data)


def _write_sensitivity_csv(path: Path, slice_data: SensitivitySlice) -> None:
    rows = [_sensitivity_record(r) for r in slice_data.records]
    write_csv_rows(rows, path, SENSITIVITY_COLUMNS)


def _sensitivity_record(record: SensitivityPerturbationRecord) -> dict[str, str]:
    return {
        "configuration": record.configuration,
        "evaluation_scope": record.evaluation_scope,
        "file_name": record.file_name,
        "perturbation": record.perturbation,
        "status": record.status,
        "rejection_reason": record.rejection_reason,
        "original_prediction_kg": format_csv_number(record.original_prediction_kg) if math.isfinite(record.original_prediction_kg) else "",
        "perturbed_prediction_kg": format_csv_number(record.perturbed_prediction_kg) if math.isfinite(record.perturbed_prediction_kg) else "",
        "delta_kg": format_csv_number(record.delta_kg) if math.isfinite(record.delta_kg) else "",
    }


def _write_eligibility_csv(path: Path, slice_data: SensitivitySlice) -> None:
    rows = [
        {
            "file_name": e.file_name,
            "status": e.status,
            "rejection_reason": e.rejection_reason,
        }
        for e in slice_data.eligibilities
    ]
    write_csv_rows(rows, path, ELIGIBILITY_COLUMNS)


def _write_demo_png(
    path: Path,
    masks: dict[str, np.ndarray],
    slice_data: SensitivitySlice,
) -> None:
    """Write a PNG showing original, contraction, and expansion for one eligible mask.

    Only visual — no predictions shown.

    Example: ``_write_demo_png(path, masks, slice_data)`` renders the demo.
    """
    import matplotlib.pyplot as plt
    from buffalo_weight.diagnostic_sensitivity_perturbations import (
        euclidean_disk,
        perturb_contraction,
        perturb_expansion,
    )
    from buffalo_weight.diagnostic_sensitivity_eligibility import (
        MORPHOLOGY_DISK_RADIUS_CANONICAL,
        CANONICAL_LONG_SIDE,
    )

    # Find the first eligible mask
    eligible_name = _first_eligible_name(slice_data)
    if eligible_name is None:
        return

    mask = masks[eligible_name]
    original_long_side = max(mask.shape)
    disk = euclidean_disk(MORPHOLOGY_DISK_RADIUS_CANONICAL, CANONICAL_LONG_SIDE, original_long_side)

    contracted = perturb_contraction(mask, disk)
    expanded = perturb_expansion(mask, disk)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original", "Contração", "Expansão"]
    images = [mask, contracted, expanded]

    for ax, title, img in zip(axes, titles, images, strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.suptitle(f"Demonstração morfológica — {eligible_name}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _first_eligible_name(slice_data: SensitivitySlice) -> str | None:
    """Return the first eligible mask name, or None if none eligible."""
    for e in slice_data.eligibilities:
        if e.status == "eligible":
            return e.file_name
    return None


def _write_report_markdown(path: Path, slice_data: SensitivitySlice) -> None:
    eligible_count = sum(1 for e in slice_data.eligibilities if e.status == "eligible")
    rejected_count = sum(1 for e in slice_data.eligibilities if e.status == "rejected")
    total_records = len(slice_data.records)
    configs = sorted({r.configuration for r in slice_data.records})

    lines = [
        "# Relatório de Diagnóstico: Sensibilidade Controlada das Predições",
        "",
        f"Total de máscaras avaliadas: {len(slice_data.eligibilities)}",
        f"Elegíveis para morfologia: {eligible_count}",
        f"Rejeitadas para morfologia: {rejected_count}",
        f"Total de registros de perturbação: {total_records}",
        f"Configurações avaliadas: {', '.join(configs)}",
        "",
        "## Resumo de Elegibilidade Morfológica",
        "",
        "| Status | Contagem |",
        "| :--- | :---: |",
        f"| Elegível | {eligible_count} |",
        f"| Rejeitada | {rejected_count} |",
        "",
    ]

    # Rejection reasons
    reasons: dict[str, int] = {}
    for e in slice_data.eligibilities:
        if e.status == "rejected" and e.rejection_reason:
            reasons[e.rejection_reason] = reasons.get(e.rejection_reason, 0) + 1
    if reasons:
        lines.extend([
            "### Motivos de Rejeição",
            "",
            "| Motivo | Contagem |",
            "| :--- | :---: |",
        ])
        for reason, count in sorted(reasons.items()):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    lines.extend([
        "## Notas",
        "",
        "- Deltas são sempre perturbado menos original.",
        "- Contração e expansão formam par inseparável: ambas ou nenhuma.",
        "- Perturbações de escala usam ±5% do foreground ao redor do centro.",
        "- Deslocamentos de 5% em cada direção nunca cortam foreground.",
        "- Perturbações sintéticas medem sensibilidade local, não substituem máscaras manuais de referência.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

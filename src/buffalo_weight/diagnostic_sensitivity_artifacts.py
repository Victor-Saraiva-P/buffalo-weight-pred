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
    target_slice: SensitivitySlice,
    masks_for_demo: dict[str, np.ndarray] | None = None,
) -> None:
    """Write tidy CSV files, demo PNG, and markdown report for sensitivity diagnostics.

    Example: ``write_sensitivity_artifacts(output_dir, target_slice)`` saves artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sensitivity_csv(output_dir / "sensitivity_perturbations.csv", target_slice)
    _write_eligibility_csv(output_dir / "morphology_eligibility.csv", target_slice)
    if masks_for_demo:
        _write_demo_png(output_dir / "morphology_demo.png", masks_for_demo, target_slice)
    _write_report_markdown(output_dir / "sensitivity_report.md", target_slice)


def _write_sensitivity_csv(path: Path, target_slice: SensitivitySlice) -> None:
    rows = [_sensitivity_record(r) for r in target_slice.records]
    write_csv_rows(rows, path, SENSITIVITY_COLUMNS)


def _sensitivity_record(record: SensitivityPerturbationRecord) -> dict[str, str]:
    orig_str = format_csv_number(record.original_prediction_kg) if math.isfinite(record.original_prediction_kg) else ""
    pert_str = format_csv_number(record.perturbed_prediction_kg) if math.isfinite(record.perturbed_prediction_kg) else ""
    delta_str = format_csv_number(record.delta_kg) if math.isfinite(record.delta_kg) else ""
    return {
        "configuration": record.configuration,
        "evaluation_scope": record.evaluation_scope,
        "file_name": record.file_name,
        "perturbation": record.perturbation,
        "status": record.status,
        "rejection_reason": record.rejection_reason,
        "original_prediction_kg": orig_str,
        "perturbed_prediction_kg": pert_str,
        "delta_kg": delta_str,
    }


def _write_eligibility_csv(path: Path, target_slice: SensitivitySlice) -> None:
    rows = [
        {"file_name": e.file_name, "status": e.status, "rejection_reason": e.rejection_reason}
        for e in target_slice.eligibilities
    ]
    write_csv_rows(rows, path, ELIGIBILITY_COLUMNS)


def _write_demo_png(
    path: Path, masks: dict[str, np.ndarray], target_slice: SensitivitySlice,
) -> None:
    """Write a PNG showing original, contraction, and expansion for one eligible mask.

    Example: ``_write_demo_png(path, masks, target_slice)`` renders the demo.
    """
    eligible_name = _first_eligible_name(target_slice)
    if eligible_name is None or eligible_name not in masks:
        return
    _render_demo_figure(path, eligible_name, masks[eligible_name])


def _render_demo_figure(path: Path, file_name: str, mask: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from buffalo_weight.diagnostic_sensitivity_eligibility import CANONICAL_LONG_SIDE, MORPHOLOGY_DISK_RADIUS_CANONICAL
    from buffalo_weight.diagnostic_sensitivity_perturbations import euclidean_disk, perturb_contraction, perturb_expansion

    disk = euclidean_disk(MORPHOLOGY_DISK_RADIUS_CANONICAL, CANONICAL_LONG_SIDE, max(mask.shape))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [mask, perturb_contraction(mask, disk), perturb_expansion(mask, disk)]
    for ax, title, img in zip(axes, ["Original", "Contração", "Expansão"], images, strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.suptitle(f"Demonstração morfológica — {file_name}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _first_eligible_name(target_slice: SensitivitySlice) -> str | None:
    """Return the first eligible mask name, or None if none eligible."""
    for e in target_slice.eligibilities:
        if e.status == "eligible":
            return e.file_name
    return None


def _write_report_markdown(path: Path, target_slice: SensitivitySlice) -> None:
    eligible_count = sum(1 for e in target_slice.eligibilities if e.status == "eligible")
    rejected_count = sum(1 for e in target_slice.eligibilities if e.status == "rejected")
    lines = _build_report_lines(target_slice, eligible_count, rejected_count)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report_lines(target_slice: SensitivitySlice, eligible: int, rejected: int) -> list[str]:
    configs = sorted({r.configuration for r in target_slice.records})
    lines = [
        "# Relatório de Diagnóstico: Sensibilidade Controlada das Predições", "",
        f"Total de máscaras avaliadas: {len(target_slice.eligibilities)}",
        f"Elegíveis para morfologia: {eligible}", f"Rejeitadas para morfologia: {rejected}",
        f"Total de registros de perturbação: {len(target_slice.records)}",
        f"Configurações avaliadas: {', '.join(configs)}", "",
        "## Resumo de Elegibilidade Morfológica", "",
        "| Status | Contagem |", "| :--- | :---: |",
        f"| Elegível | {eligible} |", f"| Rejeitada | {rejected} |", "",
    ]
    lines.extend(_report_rejection_notes(target_slice))
    return lines


def _report_rejection_notes(target_slice: SensitivitySlice) -> list[str]:
    reasons: dict[str, int] = {}
    for e in target_slice.eligibilities:
        if e.status == "rejected" and e.rejection_reason:
            reasons[e.rejection_reason] = reasons.get(e.rejection_reason, 0) + 1
    lines: list[str] = []
    if reasons:
        lines.extend(["### Motivos de Rejeição", "", "| Motivo | Contagem |", "| :--- | :---: |"])
        for reason, count in sorted(reasons.items()):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    lines.extend([
        "## Notas", "",
        "- Deltas são sempre perturbado menos original.",
        "- Contração e expansão formam par inseparável: ambas ou nenhuma.",
        "- Perturbações de escala usam ±5% do foreground ao redor do centro.",
        "- Deslocamentos de 5% que cortam o foreground são rejeitados da análise.",
        "- Perturbações sintéticas medem sensibilidade local, não substituem máscaras manuais de referência.",
    ])
    return lines

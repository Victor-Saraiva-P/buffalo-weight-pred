from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from buffalo_weight.config import load_config
from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.diagnostic_data import (
    QUALITY_FIELDS,
    audit_masks,
    domain_tests,
    feature_weight_correlations,
    nearest_visual_pairs,
    nearest_full_representation_pairs,
    integrity_tests,
    quality_error_associations,
    scale_ablation_tests,
)
from buffalo_weight.diagnostic_learning import (
    LEARNING_FIELDS,
    LearningCurveRepresentation,
    capacity_gap_rows,
    learning_curve_rows,
)
from buffalo_weight.diagnostic_metrics import (
    METRIC_FIELDS,
    bootstrap_mae,
    grouped_metrics,
    metric_summary,
    oracle_metrics,
    paired_bootstrap,
)
from buffalo_weight.diagnostic_plots import (
    plot_error_correlation,
    plot_group_mae,
    plot_learning_curve,
    plot_nearest_pairs,
    plot_quality_error,
    plot_residuals,
)
from buffalo_weight.diagnostic_robustness import segmentation_robustness
from buffalo_weight.split import read_rows
from buffalo_weight.train import format_metric


FEATURE_COLUMNS = [
    "area", "perimeter", "solidity", "circularity", "equivalent_diameter", "bbox_width",
    "bbox_height", "bbox_area", "aspect_ratio", "extent", "convex_area", "convexity",
    "major_axis_length", "minor_axis_length", "hu_moment_1", "hu_moment_2",
]
MODEL_NAMES = [
    "dual_pca24_canonical16", "tuned_96_pca24", "geometry_resnet18_pretrained_last_block",
    "fusion_original_stretch_log", "random_forest_baseline", "resnet18_pretrained_last_block",
]


def _write(rows: list[dict[str, str]], path: Path, preferred: list[str] | None = None) -> None:
    fields = list(preferred or [])
    fields.extend(sorted({key for row in rows for key in row if key not in fields}))
    write_csv_rows(rows, path, fields)


def _prediction_vectors(
    train_dir: Path, rows: list[dict[str, str]], model_names: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, str]]]:
    names = [row["file_name"] for row in rows]
    predictions = {}
    reference = {}
    for model_name in model_names:
        indexed = {row["file_name"]: row for row in read_rows(train_dir / model_name / "predictions.csv")}
        predictions[model_name] = np.asarray([float(indexed[name]["y_pred"]) for name in names])
        if not reference:
            reference = indexed
    actual = np.asarray([float(row["weight"]) for row in rows])
    return actual, predictions, reference


def _weight_quintiles(actual: np.ndarray) -> list[str]:
    order = np.argsort(np.argsort(actual))
    return [f"Q{min(int(rank * 5 / len(actual)) + 1, 5)}" for rank in order]


def _conditional_rows(
    rows: list[dict[str, str]], audit_rows: list[dict[str, str]], reference: dict[str, dict[str, str]],
    actual: np.ndarray, predicted: np.ndarray,
) -> list[dict[str, str]]:
    resolution = {row["file_name"]: row["resolution"] for row in audit_rows}
    metrics = [{"group_type": "global", "group": "all", **metric_summary(actual, predicted)}]
    metrics.extend(grouped_metrics(actual, predicted, [row["farm"] for row in rows], "farm"))
    metrics.extend(grouped_metrics(actual, predicted, [reference[row["file_name"]]["weight_category"] for row in rows], "weight_category"))
    metrics.extend(grouped_metrics(actual, predicted, [resolution[row["file_name"]] for row in rows], "resolution"))
    metrics.extend(grouped_metrics(actual, predicted, _weight_quintiles(actual), "weight_quintile"))
    return metrics


def _model_rows(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, str]]:
    return [{"model": name, **metric_summary(actual, predicted)} for name, predicted in predictions.items()]


def _bootstrap_rows(actual: np.ndarray, predictions: dict[str, np.ndarray], ensemble: np.ndarray) -> list[dict[str, str]]:
    candidate = predictions["dual_pca24_canonical16"]
    rows = [{"comparison": "canonical_mae", **bootstrap_mae(actual, candidate)}]
    for name, reference in (
        ("canonical_vs_fusion_log", predictions["fusion_original_stretch_log"]),
        ("canonical_vs_ensemble", ensemble),
    ):
        rows.append({"comparison": name, **paired_bootstrap(actual, candidate, reference)})
    return rows


def _error_outputs(
    actual: np.ndarray, predictions: dict[str, np.ndarray], output_dir: Path
) -> tuple[dict[str, str], np.ndarray]:
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    errors = matrix - actual[:, None]
    correlation = np.corrcoef(errors, rowvar=False)
    _write(
        [{"model": name, **{other: format_metric(float(correlation[i, j])) for j, other in enumerate(names)}} for i, name in enumerate(names)],
        output_dir / "error_correlation.csv",
    )
    plot_error_correlation(names, correlation, output_dir / "plots" / "error_correlation.png")
    return oracle_metrics(actual, matrix), correlation


def _learning_summary(run_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries = []
    for fraction in sorted({float(row["fraction"]) for row in run_rows}):
        selected = [row for row in run_rows if float(row["fraction"]) == fraction]
        summaries.append({
            "fraction": str(fraction),
            "n_train_mean": format_metric(float(np.mean([float(row["n_train"]) for row in selected]))),
            **_mean_std(selected, "train_mae", "train_mae"),
            **_mean_std(selected, "validation_mae", "validation_mae"),
            **_mean_std(selected, "heavy_validation_mae", "heavy_mae"),
        })
    return summaries


def _learning_tests(run_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed = {(row["split_seed"], row["fold"], row["fraction"]): row for row in run_rows}
    rng = np.random.default_rng(42)
    results = []
    for start, end in (("0.25", "1.0"), ("0.75", "1.0")):
        keys = sorted((seed, fold) for seed, fold, fraction in indexed if fraction == start)
        improvements = np.asarray([
            float(indexed[seed, fold, start]["validation_mae"]) - float(indexed[seed, fold, end]["validation_mae"])
            for seed, fold in keys
        ])
        samples = improvements[rng.integers(0, len(improvements), size=(10000, len(improvements)))].mean(axis=1)
        results.append({
            "comparison": f"fraction_{start}_to_{end}",
            "mae_improvement": format_metric(float(improvements.mean())),
            "ci_low": format_metric(float(np.quantile(samples, 0.025))),
            "ci_high": format_metric(float(np.quantile(samples, 0.975))),
            "fraction_runs_improved": format_metric(float((improvements > 0).mean())),
        })
    return results


def _capacity_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries = []
    for leaf in sorted({row["min_samples_leaf"] for row in rows}):
        selected = [row for row in rows if row["min_samples_leaf"] == leaf]
        summaries.append({"min_samples_leaf": leaf, **_mean_std(selected, "train_mae", "train_mae"), **_mean_std(selected, "validation_mae", "validation_mae")})
    return summaries


def _quality_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    foreground = np.asarray([float(row["foreground_ratio"]) for row in rows])
    return {
        "mask_count": str(len(rows)),
        "multiple_component_count": str(sum(int(row["component_count"]) > 1 for row in rows)),
        "hole_count": str(sum(float(row["hole_ratio"]) > 0 for row in rows)),
        "border_touch_count": str(sum(row["touches_border"] == "1" for row in rows)),
        "foreground_ratio_min": format_metric(float(foreground.min())),
        "foreground_ratio_median": format_metric(float(np.median(foreground))),
        "foreground_ratio_max": format_metric(float(foreground.max())),
    }


def _mean_std(rows: list[dict[str, str]], field: str, prefix: str) -> dict[str, str]:
    values = np.asarray([float(row[field]) for row in rows])
    return {f"{prefix}_mean": format_metric(float(values.mean())), f"{prefix}_std": format_metric(float(values.std()))}


def _report(
    output_dir: Path, conditional: list[dict[str, str]], learning: list[dict[str, str]], domain: list[dict[str, str]],
    bootstrap: list[dict[str, str]], oracle: dict[str, str], quality: list[dict[str, str]],
    robustness: list[dict[str, str]], pairs: list[dict[str, str]], learning_tests: list[dict[str, str]],
    capacity: list[dict[str, str]], integrity: list[dict[str, str]], scale_ablation: list[dict[str, str]],
    neighbor_summary: dict[str, str], quality_summary: dict[str, str],
) -> None:
    b10 = next(row for row in conditional if row["group_type"] == "weight_category" and row["group"] == "B10")
    farm = next(row for row in domain if row["test"] == "farm_common_support_shape")
    first, last = learning[0], learning[-1]
    strongest_quality = max(quality, key=lambda row: abs(float(row["spearman_abs_error"])))
    worst_perturbation = max(robustness, key=lambda row: float(row["mae"]))
    lines = _report_lines(
        b10, farm, first, last, domain, bootstrap, oracle, strongest_quality, worst_perturbation,
        pairs[0], learning_tests, capacity, integrity, scale_ablation, neighbor_summary, quality_summary,
    )
    output_dir.joinpath("report.md").write_text("\n".join(lines))


def _report_lines(
    b10: dict[str, str], farm: dict[str, str], first: dict[str, str], last: dict[str, str],
    domain: list[dict[str, str]], bootstrap: list[dict[str, str]], oracle: dict[str, str], quality: dict[str, str],
    perturbation: dict[str, str], pair: dict[str, str], learning_tests: list[dict[str, str]],
    capacity: list[dict[str, str]], integrity: list[dict[str, str]], scale_ablation: list[dict[str, str]],
    neighbor_summary: dict[str, str], quality_summary: dict[str, str],
) -> list[str]:
    ci = bootstrap[0]
    paired = bootstrap[1]
    final_slope = learning_tests[1]
    interpolating = next(row for row in capacity if row["min_samples_leaf"] == "1")
    maximum_iou = next(row["value"] for row in integrity if row["test"] == "maximum_canonical_iou")
    scale = {row["representation"]: row["mae"] for row in scale_ablation}
    farm_all = next(row for row in domain if row["test"] == "farm_all_features")
    farm_weight = next(row for row in domain if row["test"] == "farm_weight_only")
    resolution = next(row for row in domain if row["test"] == "resolution_all_features")
    transfer = [row for row in domain if row["metric"] == "mae"]
    return [
        "# Diagnóstico empírico do limite de MAE", "",
        "## Evidências observadas", "",
        f"- Melhor modelo individual: MAE {ci['mae']} kg, IC95% bootstrap {ci['ci_low']}–{ci['ci_high']} kg.",
        f"- Faixa B10: n={b10['n']}, MAE {b10['mae']} kg e viés {b10['bias']} kg.",
        f"- Learning curve: validação caiu de {first['validation_mae_mean']} para {last['validation_mae_mean']} kg ao passar de {first['n_train_mean']} para {last['n_train_mean']} amostras; treino final {last['train_mae_mean']} kg.",
        f"- Nos 20% mais pesados, o MAE final foi {last['heavy_mae_mean']} kg.",
        f"- De 75% para 100% dos dados, o ganho pareado foi {final_slope['mae_improvement']} kg (IC95% {final_slope['ci_low']}–{final_slope['ci_high']}).",
        f"- Com folhas unitárias, treino chegou a {interpolating['train_mae_mean']} kg e validação ficou em {interpolating['validation_mae_mean']} kg.",
        f"- Fazenda: balanced accuracy {farm_all['balanced_accuracy_mean']} com features e {farm_weight['balanced_accuracy_mean']} só com peso; no suporte comum usando apenas forma, {farm['balanced_accuracy_mean']} (nível de acaso ≈ 0,5).",
        f"- Resolução é classificável pelas features com balanced accuracy {resolution['balanced_accuracy_mean']}; transferência entre fazendas no suporte comum teve MAE {transfer[0]['mae']} e {transfer[1]['mae']} kg.",
        f"- Ablation OOF: escala+forma {scale['all_features']} kg, somente forma {scale['shape_only']} kg, somente área {scale['area_only']} kg.",
        f"- Oracle entre modelos atuais: MAE {oracle['oracle_mae']} kg; existe complementaridade, mas não elimina o erro.",
        f"- Melhor associação morfológica com erro: {quality['metric']}, Spearman {quality['spearman_abs_error']} (p={quality['p_value']}).",
        f"- Auditoria bruta: {quality_summary['multiple_component_count']} máscaras com múltiplos componentes, {quality_summary['border_touch_count']} tocando borda e {quality_summary['hole_count']} com algum buraco.",
        f"- Pior perturbação sintética: {perturbation['scenario']}, MAE {perturbation['mae']} kg e mudança média de predição {perturbation['mean_prediction_change']} kg.",
        f"- Par visual contraditório: IoU canônica {pair['canonical_iou']}, diferença de peso {pair['weight_difference']} kg.",
        f"- Não há par canônico com IoU ≥ 0,95; maior IoU observado {maximum_iou}.",
        f"- No espaço completo do modelo, a diferença mediana de peso para o vizinho mais próximo foi {neighbor_summary['median_nearest_weight_difference']} kg e P90 {neighbor_summary['p90_nearest_weight_difference']} kg.",
        f"- Canônico versus fusão log: delta pareado {paired['mae_delta']} kg, IC95% {paired['ci_low']}–{paired['ci_high']} kg.",
        "", "## Interpretação limitada às evidências", "",
        "- MAE zero não é compatível com a generalização observada: há grande gap treino-validação, erros extremos sistemáticos e máscaras semelhantes associadas a pesos diferentes.",
        "- A learning curve mede se mais amostras ainda reduzem erro; ela não prova quanto um futuro lote reduzirá o MAE.",
        "- Classificação de fazenda e transferência medem domínio de aquisição, mas fazenda e peso estão confundidos nos dados atuais.",
        "- Perturbações sintéticas medem sensibilidade local, não substituem máscaras manuais de referência.",
        "", "## Não testável com o conjunto atual", "",
        "- Ruído da balança e repetibilidade do peso, pois não há pesagens repetidas.",
        "- Erro real da segmentação, pois não há máscaras manuais ground truth.",
        "- Generalização prospectiva para novos animais, sessões, câmeras ou fazendas.",
        "- Ausência de fotos repetidas do mesmo animal, pois não há ID independente de animal/sessão.",
    ]


def run_diagnostics(shared_config_path: Path, output_dir: Path, learning_seed_count: int) -> None:
    """Run the empirical error investigation; for example, ``run_diagnostics(config, output, 5)``."""
    config = load_config(shared_config_path)
    features = config["features"]
    data = config["data"]
    training = config["training"]
    if not isinstance(features, dict) or not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("shared config sections were not maps; expected data, features and training maps")
    rows = read_rows(Path(str(features["features_index_path"])))
    masks_dir = Path(str(data["masks_dir"]))
    train_dir = Path(str(training["output_dir"]))
    actual, predictions, reference = _prediction_vectors(train_dir, rows, MODEL_NAMES)
    ensemble = np.mean(np.column_stack([predictions[name] for name in MODEL_NAMES[:3]]), axis=1)
    audit = audit_masks(masks_dir, rows)
    conditional = _conditional_rows(rows, audit, reference, actual, predictions[MODEL_NAMES[0]])
    errors = {row["file_name"]: abs(predictions[MODEL_NAMES[0]][index] - actual[index]) for index, row in enumerate(rows)}
    quality = quality_error_associations(audit, errors)
    quality_summary = _quality_summary(audit)
    domain = domain_tests(rows, FEATURE_COLUMNS, audit)
    pairs, canonical_masks = nearest_visual_pairs(masks_dir, rows)
    full_pairs, neighbor_summary = nearest_full_representation_pairs(masks_dir, rows, FEATURE_COLUMNS)
    learning_representation = LearningCurveRepresentation(rows, FEATURE_COLUMNS, masks_dir, 96)
    learning_runs = learning_curve_rows(rows, learning_representation, list(range(learning_seed_count)))
    learning = _learning_summary(learning_runs)
    learning_tests = _learning_tests(learning_runs)
    capacity_runs = capacity_gap_rows(rows, learning_representation, list(range(learning_seed_count)))
    capacity = _capacity_summary(capacity_runs)
    robustness_rows = [dict(row) for row in read_rows(Path(str(config["split"]["split_path"])))]
    robustness = segmentation_robustness(masks_dir, robustness_rows, FEATURE_COLUMNS)
    bootstrap = _bootstrap_rows(actual, predictions, ensemble)
    integrity = integrity_tests(rows, audit, canonical_masks)
    scale_ablation = scale_ablation_tests(rows, FEATURE_COLUMNS)
    oracle, _ = _error_outputs(actual, predictions, output_dir)
    _write(conditional, output_dir / "conditional_metrics.csv", METRIC_FIELDS)
    _write(_model_rows(actual, predictions), output_dir / "model_metrics.csv")
    _write(bootstrap, output_dir / "bootstrap_comparisons.csv")
    _write(audit, output_dir / "mask_quality.csv", QUALITY_FIELDS)
    _write(quality, output_dir / "mask_quality_error_associations.csv")
    _write([quality_summary], output_dir / "mask_quality_summary.csv")
    _write(domain, output_dir / "domain_tests.csv")
    _write(feature_weight_correlations(rows, "area"), output_dir / "area_weight_correlations.csv")
    _write(pairs, output_dir / "nearest_visual_contradictions.csv")
    _write(full_pairs, output_dir / "nearest_full_representation_contradictions.csv")
    _write([neighbor_summary], output_dir / "nearest_full_representation_summary.csv")
    _write(learning_runs, output_dir / "learning_curve_runs.csv", LEARNING_FIELDS)
    _write(learning, output_dir / "learning_curve_summary.csv")
    _write(learning_tests, output_dir / "learning_curve_tests.csv")
    _write(capacity_runs, output_dir / "capacity_gap_runs.csv")
    _write(capacity, output_dir / "capacity_gap_summary.csv")
    _write(robustness, output_dir / "segmentation_robustness.csv")
    _write([oracle], output_dir / "oracle_summary.csv")
    _write(integrity, output_dir / "integrity_tests.csv")
    _write(scale_ablation, output_dir / "scale_ablation.csv")
    plot_residuals(actual, predictions[MODEL_NAMES[0]], [row["farm"] for row in rows], output_dir / "plots" / "residuals.png")
    plot_group_mae(conditional, "weight_category", output_dir / "plots" / "mae_by_weight_category.png")
    plot_learning_curve(learning, output_dir / "plots" / "learning_curve.png")
    plot_quality_error(audit, errors, output_dir / "plots" / "mask_quality_vs_error.png")
    plot_nearest_pairs(pairs, canonical_masks, {row["file_name"]: i for i, row in enumerate(rows)}, output_dir / "plots" / "nearest_visual_pairs.png")
    _report(
        output_dir, conditional, learning, domain, bootstrap, oracle, quality, robustness,
        pairs, learning_tests, capacity, integrity, scale_ablation, neighbor_summary, quality_summary,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--output-dir", default="generated/diagnostics")
    parser.add_argument("--learning-seed-count", type=int, default=5)
    args = parser.parse_args(argv)
    run_diagnostics(Path(args.shared_config), Path(args.output_dir), args.learning_seed_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from buffalo_weight.config import load_config


CATEGORY_LABELS = {
    "Q1": "Leves",
    "Q2": "Medio-Leves",
    "Q3": "Medio-Pesados",
    "Q4": "Pesados",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def parse_int(value: object, name: str) -> int:
    try:
        return int(str(value))
    except ValueError as error:
        raise ValueError(f"config {name} must be an integer") from error


def parse_weight(value: str, file_name: str) -> float:
    try:
        return float(value.replace(",", "."))
    except ValueError as error:
        raise ValueError(f"Invalid weight for {file_name}: {value}") from error


def assign_weight_categories(rows: list[dict[str, str]]) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            parse_weight(row["weight"], row.get("file_name", "")),
            row.get("file_name", ""),
        ),
    )
    total = len(sorted_rows)
    for index, row in enumerate(sorted_rows):
        category_index = min((index * 4) // total + 1, 4)
        category = f"Q{category_index}"
        row["weight_category"] = category
        row["weight_category_label"] = CATEGORY_LABELS[category]


def assign_folds(rows: list[dict[str, str]], k: int, random_state: int) -> None:
    labels = [row["weight_category"] for row in rows]
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for fold, (_, validation_indexes) in enumerate(splitter.split(rows, labels), start=1):
        for index in validation_indexes:
            rows[index]["fold"] = str(fold)


def write_split(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "file_name",
        "farm",
        "weight",
        "tag",
        "weight_category",
        "weight_category_label",
        "fold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temp_path.replace(path)


def write_distribution_data(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = ["file_name", "farm", "weight", "weight_category", "weight_category_label"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temp_path.replace(path)


def plot_weight_distribution(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    x_by_category = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    x_values = []
    y_values = []
    for row in rows:
        file_hash = sum(ord(char) for char in row.get("file_name", ""))
        jitter = ((file_hash % 21) - 10) / 100
        x_values.append(x_by_category[row["weight_category"]] + jitter)
        y_values.append(parse_weight(row["weight"], row.get("file_name", "")))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, color="#1f77b4", alpha=0.75, edgecolors="none")
    ax.set_xticks([1, 2, 3, 4], ["Q1\nLeves", "Q2\nMedio-Leves", "Q3\nMedio-Pesados", "Q4\nPesados"])
    ax.set_ylabel("Peso (kg)")
    ax.set_xlabel("Categoria de Peso")
    ax.set_title("Distribuicao de pesos por Categoria de Peso")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_split(config_path: Path) -> None:
    config = load_config(config_path)
    output = config["output"]
    split = config["split"]
    if not isinstance(output, dict):
        raise ValueError("config output section must be a map")
    if not isinstance(split, dict):
        raise ValueError("config split section must be a map")

    rows = read_rows(Path(str(output["features_index_path"])))
    if not rows:
        raise ValueError("feature index is empty")

    k = parse_int(split["k"], "split.k")
    random_state = parse_int(split["random_state"], "split.random_state")
    assign_weight_categories(rows)
    assign_folds(rows, k, random_state)
    write_split(rows, Path(str(split["split_path"])))
    write_distribution_data(rows, Path(str(split["distribution_plot_data_path"])))
    plot_weight_distribution(rows, Path(str(split["distribution_plot_path"])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    try:
        generate_split(Path(args.config))
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

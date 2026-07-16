from __future__ import annotations

import argparse
import json
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(project_root: Path) -> dict:
    plot_path = project_root / "figures" / "predictions" / "rdkit_notebook_predicted_vs_actual_logkp.png"
    rdkit_python = Path("/opt/anaconda3/envs/rdkit_env/bin/python")

    cells = [
        markdown_cell(
            "# RDKit-to-LogKp Prediction Demo\n\n"
            "This notebook shows the full single-compound prediction path for a new SMILES string.\n\n"
            "- RDKit is used for SMILES validation and to compute `logP` plus a small descriptor summary.\n"
            "- The trained models still expect the CDK-style feature panel used during training, so the notebook reuses the project's descriptor generator to build the exact model input vector.\n"
            "- The final section predicts `logKp`, computes the formula-based reference `logKp`, and plots predicted vs actual.\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "\n"
            f"PROJECT_ROOT = Path({project_root.as_posix()!r})\n"
            "SRC = PROJECT_ROOT / 'src'\n"
            "if str(SRC) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC))\n"
            "\n"
            "from skin_permeation.data_loading import load_bundle\n"
            "from skin_permeation.formulas import calculate_formula_logkp\n"
            "from skin_permeation.paths import ProjectPaths\n"
            "from skin_permeation.prediction import (\n"
            "    align_descriptor_frame_to_feature_columns,\n"
            "    build_inference_fill_values,\n"
            "    compute_rdkit_properties,\n"
            "    feature_columns_from_training_frame,\n"
            "    generate_training_compatible_descriptors,\n"
            "    predict_logkp_from_smiles,\n"
            ")\n"
            "\n"
            "paths = ProjectPaths.discover(PROJECT_ROOT)\n"
            "bundle = load_bundle(paths)\n"
            f"RDKIT_PYTHON = {rdkit_python.as_posix()!r} if Path({rdkit_python.as_posix()!r}).exists() else None\n"
        ),
        code_cell(
            "sample_smiles = 'CCO'\n"
            "texpi = 310.0\n"
            "sample_smiles, texpi\n"
        ),
        markdown_cell(
            "## 1. Validate the SMILES and compute RDKit descriptors\n\n"
            "The helper below works with the current Python if `rdkit` is installed there, or falls back to the configured RDKit environment."
        ),
        code_cell(
            "rdkit_result = compute_rdkit_properties(sample_smiles, python_executable=RDKIT_PYTHON)\n"
            "pd.Series(\n"
            "    {\n"
            "        'Input SMILES': rdkit_result.input_smiles,\n"
            "        'Canonical SMILES': rdkit_result.canonical_smiles,\n"
            "        'RDKit logP': rdkit_result.logp,\n"
            "        'RDKit molecular weight': rdkit_result.molecular_weight,\n"
            "        **rdkit_result.descriptor_summary,\n"
            "    }\n"
            ").to_frame('value')\n"
        ),
        markdown_cell(
            "## 2. Build the exact model feature vector\n\n"
            "The project models were trained on the cleaned CDK-style descriptor table in `clean_trial4.csv`, so we generate the compatible descriptor row first and then align it to the 146-feature training schema."
        ),
        code_cell(
            "raw_descriptors = generate_training_compatible_descriptors(\n"
            "    paths=paths,\n"
            "    smiles=rdkit_result.canonical_smiles,\n"
            "    texpi=texpi,\n"
            ")\n"
            "feature_columns = feature_columns_from_training_frame(bundle.clean_trial4)\n"
            "fill_values = build_inference_fill_values(bundle)\n"
            "feature_frame = align_descriptor_frame_to_feature_columns(\n"
            "    descriptor_frame=raw_descriptors,\n"
            "    feature_columns=feature_columns,\n"
            "    fill_values=fill_values,\n"
            "    texpi=texpi,\n"
            ")\n"
            "print(f'Feature vector shape: {feature_frame.shape}')\n"
            "feature_frame.iloc[:, :20].T.rename(columns={feature_frame.index[0]: 'value'})\n"
        ),
        markdown_cell(
            "## 3. Compute the formula-based reference `logKp`\n\n"
            "This project now exposes the Potts-Guy style formula consistently through `calculate_formula_logkp(logP, molecular_weight)`."
        ),
        code_cell(
            "formula_logkp = calculate_formula_logkp(rdkit_result.logp, rdkit_result.molecular_weight)\n"
            "formula_logkp\n"
        ),
        markdown_cell(
            "## 4. Predict `logKp` with the existing trained models\n\n"
            "The high-level helper below reuses the same inference pipeline used by the runnable script."
        ),
        code_cell(
            f"plot_output = PROJECT_ROOT / {str(plot_path.relative_to(project_root)).replace('\\\\', '/')!r}\n"
            "prediction_result = predict_logkp_from_smiles(\n"
            "    smiles=sample_smiles,\n"
            "    paths=paths,\n"
            "    texpi=texpi,\n"
            "    rdkit_python=RDKIT_PYTHON,\n"
            "    plot_output=plot_output,\n"
            ")\n"
            "prediction_result.predictions\n"
        ),
        markdown_cell("## 5. Visualize predicted vs actual `logKp`"),
        code_cell(
            "display(Image(filename=str(prediction_result.plot_path)))\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the RDKit integration notebook.")
    parser.add_argument("output", help="Notebook path to write.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root used inside the notebook setup cells.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(Path(args.project_root).resolve())
    output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

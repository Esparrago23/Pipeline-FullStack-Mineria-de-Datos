from __future__ import annotations

import json
from typing import Any

from .analysis_preprocessing import run_analysis_preprocessing
from .modeling import run_modeling
from .warehouse import run_warehouse


def run_pipeline() -> dict[str, Any]:
    analysis = run_analysis_preprocessing()
    warehouse = run_warehouse()
    modeling = run_modeling()
    return {
        "status": "ok",
        "analysis": {
            "raw": analysis["raw"],
            "processed": analysis["processed"],
            "split_shapes": analysis["split_shapes"],
        },
        "warehouse": {
            "duckdb_path": warehouse["duckdb_path"],
            "tables": warehouse["tables"],
            "cube_validation": warehouse["cube_validation"],
        },
        "modeling": {
            "classification_best_model": modeling["classification"]["best_model"],
            "regression_best_model": modeling["regression"]["best_model"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))

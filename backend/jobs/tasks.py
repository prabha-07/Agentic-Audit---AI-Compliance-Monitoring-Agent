from __future__ import annotations

import json
import os
from pathlib import Path

from backend.api.routes import _compute_evaluation, _latest_pipeline_log_path
from backend.graph import run_pipeline
from backend.jobs.queue import set_job_status


def process_analysis_job(
    job_id: str,
    input_path: str,
    filename: str,
    ground_truth: dict | None,
    ground_truth_source: dict | None,
) -> dict:
    set_job_status(job_id, {"status": "running"})
    try:
        state = run_pipeline(input_path)
        doc_id = state["doc_id"]
        vr = state.get("violation_report", {})
        evaluation = _compute_evaluation(state, ground_truth, ground_truth_source)

        project_root = Path(__file__).resolve().parents[2]
        eval_path = project_root / "outputs" / "reports" / doc_id / "raw" / "evaluation.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as fh:
            json.dump(evaluation, fh, indent=2)

        log_path = _latest_pipeline_log_path(doc_id)
        result = {
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": state.get("doc_type"),
            "regulations": state.get("regulation_scope", []),
            "risk_score": state.get("risk_score"),
            "risk_level": state.get("risk_level"),
            "articles_evaluated": vr.get("articles_evaluated"),
            "hallucination_rate": vr.get("hallucination_rate"),
            "evaluation": evaluation,
            "evaluation_url": f"/api/v1/reports/{doc_id}/evaluation",
            "violation_report_url": f"/api/v1/reports/{doc_id}",
            "assessment_report_url": f"/api/v1/reports/{doc_id}/assessment",
            "remediation_report_url": f"/api/v1/reports/{doc_id}/remediation",
            "pipeline_log_url": f"/api/v1/reports/{doc_id}/pipeline-log"
            if log_path and log_path.exists()
            else None,
        }
        set_job_status(job_id, {"status": "completed", "result": result})
        return result
    except Exception as e:  # noqa: BLE001
        set_job_status(job_id, {"status": "failed", "error": str(e)})
        raise
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass

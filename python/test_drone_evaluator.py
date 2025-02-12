import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from drone_evaluator import DroneRacerEvaluator


def test_evaluate_baseline_1():
    evaluator = DroneRacerEvaluator()
    client_payload = {
        "submission_file_path": "sample_models/dqn-agent-1.safetensors",
        "aicrowd_submission_id": 1123,
        "aicrowd_participant_id": 1234
    }

    result = evaluator._evaluate(client_payload)
    assert "score" in result
    assert "score_secondary" in result
    assert "media_video_path" in result
    assert isinstance(result["score"], (float, np.float64))
    assert np.isclose(result["score"], -75.29999999999993)
    assert np.isclose(result["score_secondary"], 14.297272467152291)

def test_evaluate_baseline_2():
    evaluator = DroneRacerEvaluator()
    client_payload = {
        "submission_file_path": "sample_models/dqn-agent-2.safetensors",
        "aicrowd_submission_id": 1123,
        "aicrowd_participant_id": 1234
    }

    result = evaluator._evaluate(client_payload)
    assert "score" in result
    assert "score_secondary" in result
    assert "media_video_path" in result
    assert isinstance(result["score"], (float, np.float64))
    assert np.isclose(result["score"], -73.26000000000008)
    assert np.isclose(result["score_secondary"], 6.620604202034626)
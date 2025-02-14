import numpy as np
import pytest
from drone_evaluator import DroneRacerEvaluator
from pprint import pprint
TEST_CASES = [
    ("sample_models/dqn-agent-1.safetensors", -112.16999999999965, 23.66212374238602),
    ("sample_models/dqn-agent-2.safetensors", -277.9899999999999, 23.618063002710592),
    ("sample_models/dqn-agent-3.safetensors", -114.7099999999997, 7.361175177918399),
    ("sample_models/dqn-agent-4.safetensors", -161.50999999999974, 15.258535316340211),
    ("sample_models/dqn-agent-5.safetensors", -190.30999999999975, 10.718997154584919),
]

@pytest.mark.parametrize("model_path,expected_score,expected_secondary", TEST_CASES)
def test_evaluate_baseline(model_path, expected_score, expected_secondary):
    evaluator = DroneRacerEvaluator()
    client_payload = {
        "submission_file_path": model_path,
        "aicrowd_submission_id": 1123,
        "aicrowd_participant_id": 1234
    }

    pprint(client_payload)
    result = evaluator._evaluate(client_payload)
    pprint(result)

    assert "score" in result
    assert "score_secondary" in result
    assert "media_video_path" in result
    assert isinstance(result["score"], (float, np.float64))
    assert np.isclose(result["score"], expected_score)
    assert np.isclose(result["score_secondary"], expected_secondary)


if __name__ == "__main__":
    for test_case in TEST_CASES:
        test_evaluate_baseline(test_case[0], test_case[1], test_case[2])

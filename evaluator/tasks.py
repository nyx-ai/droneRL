from drone_evaluator import DroneRacerEvaluator

def evaluate_submission(client_payload):
    evaluator = DroneRacerEvaluator(answer_folder_path=".")
    result = evaluator._evaluate(client_payload)
    return result 
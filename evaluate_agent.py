import argparse
from drone_evaluator import DroneRacerEvaluator
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description='Evaluate drone racing models')
    parser.add_argument('model_paths', type=str, nargs='+', help='Path(s) to model file(s) (.safetensors)')
    args = parser.parse_args()

    evaluator = DroneRacerEvaluator()
    
    for i, model_path in enumerate(args.model_paths, 1):
        print(f"\n{'='*50}")
        print(f"Evaluating model {i}/{len(args.model_paths)}: {model_path}")
        print(f"{'='*50}\n")
        
        client_payload = {
            "submission_file_path": model_path,
            "aicrowd_submission_id": 0,
            "aicrowd_participant_id": 0
        }

        result = evaluator._evaluate(client_payload)
        pprint(result)

if __name__ == "__main__":
    main() 
import os.path

import numpy as np
import torch
import tqdm

from python.env.env import DeliveryDrones as DeliveryDrones
from python.env.wrappers import WindowedGridView as WindowedGridView
from python.helpers.rl_helpers import set_seed
from PIL import Image
import tempfile
import aicrowd_helpers
from safetensors.torch import load_file
from python.agents.dqn import DenseQNetwork


class DroneRacerEvaluator:
    def __init__(self, answer_folder_path=".", round=1):
        """
        `round` : Holds the round for which the evaluation is being done.
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_folder_path = answer_folder_path
        self.round = round

        ################################################
        ################################################
        # Evaluation State Variables
        ################################################
        self.EPISODE_SEEDS = [845, 99, 65, 96, 85, 39, 51, 17, 52, 35]
        self.TOTAL_EPISODE_STEPS = 1000
        self.participating_agents = {
            "baseline-1": "baseline_models/dqn-agent-1.safetensors",
            "baseline-2": "baseline_models/dqn-agent-2.safetensors",
            "baseline-3": "baseline_models/dqn-agent-3.safetensors",
            "baseline-4": "baseline_models/dqn-agent-4.safetensors",
            "baseline-5": "baseline_models/dqn-agent-5.safetensors",
        }

        self.video_directory_path = tempfile.mkdtemp()

        ################################################
        ################################################
        # Helper Functions
        ################################################

    def agent_id(self, agent_name):
        """
        Returns a unique numeric id for an agent_name
        """
        agent_names = sorted(self.participating_agents.keys())
        return agent_names.index(agent_name)

    def agent_name_from_id(self, agent_id):
        """
        Returns the unique agent name from an agent_id
        """
        agent_names = sorted(self.participating_agents.keys())
        return agent_names[agent_id]

    def get_agent_name_mapping(self):
        agent_names = sorted(self.participating_agents.keys())
        _agent_name_mapping = {}
        for _agent_name in agent_names:
            _agent_id = self.agent_id(_agent_name)
            _agent_name_mapping[_agent_id] = _agent_name
        return _agent_name_mapping

    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]

        self.video_directory_path = tempfile.mkdtemp()
        print("Video Directory Path : ", self.video_directory_path)

        ################################################
        ################################################
        # Load submission model
        ################################################

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        
        # Create environment
        # (env is already created below: env = WindowedGridView(...))
        
        env_params = {  # Updates to the default params have to be added after this instantiation
            'charge_reward': -0.1,
            'crash_reward': -1,
            'delivery_reward': 1,
            'charge': 20,
            'discharge': 10,
            'drone_density': 0.05,
            'dropzones_factor': 2,
            'n_drones': 3,
            'packets_factor': 3,
            'pickup_reward': 0,
            'rgb_render_rescale': 1.0,
            'skyscrapers_factor': 3,
            'stations_factor': 2
        }
        env_params["n_drones"] = len(self.participating_agents.keys())
        env_params["rgb_render_rescale"] = 2.0  # large video - florian's request

        env = WindowedGridView(DeliveryDrones(env_params), radius=3)
        # Load baseline agents using safetensors
        baseline_agents = {}
        for agent_name, agent_path in self.participating_agents.items():
            if agent_name != "YOU":
                full_path = os.path.join(self.answer_folder_path, agent_path)
                loaded = load_file(full_path)
                
                # Extract metadata and state dict
                metadata = {}
                state_dict = {}
                for key, value in loaded.items():
                    if key.startswith("_metadata_"):
                        metadata[key[10:]] = value
                    else:
                        state_dict[key] = value
                
                # Create appropriate network based on metadata
                if metadata["network_type"] == "dense":
                    hidden_layers = eval(metadata["hidden_layers"])
                    net = DenseQNetwork(env, hidden_layers=hidden_layers)
                else:
                    conv_layers = eval(metadata["conv_layers"])
                    net = ConvQNetwork(env, conv_layers=conv_layers)
                
                # Verify dimensions match
                assert net.input_size == int(metadata["input_size"]), f"Input size mismatch for {agent_name}"
                assert net.output_size == int(metadata["output_size"]), f"Output size mismatch for {agent_name}"
                
                net.load_state_dict(state_dict)
                baseline_agents[agent_name] = net

        # Load submission agent
        loaded = load_file(submission_file_path)
        
        # Extract metadata and state dict
        metadata = {}
        state_dict = {}
        for key, value in loaded.items():
            if key.startswith("_metadata_"):
                metadata[key[10:]] = value
            else:
                state_dict[key] = value
        
        # Create appropriate network based on metadata
        if metadata["network_type"] == "dense":
            hidden_layers = eval(metadata["hidden_layers"])
            submission_model = DenseQNetwork(env, hidden_layers=hidden_layers)
        else:
            conv_layers = eval(metadata["conv_layers"])
            submission_model = ConvQNetwork(env, conv_layers=conv_layers)
        
        # Verify dimensions match
        assert submission_model.input_size == int(metadata["input_size"]), "Submission input size mismatch"
        assert submission_model.output_size == int(metadata["output_size"]), "Submission output size mismatch"
        
        submission_model.load_state_dict(state_dict)
        self.participating_agents["YOU"] = submission_model

        self.overall_scores = []

        for _episode_idx, episode_seed in enumerate(self.EPISODE_SEEDS):
            ################################################
            ################################################
            # Run Episode
            ################################################
            episode_scores = np.zeros(len(self.participating_agents.keys()))

            ################################################
            ################################################
            # Env Instantiation
            ################################################
            set_seed(env, episode_seed)  # Seed

            agent_name_mappings = self.get_agent_name_mapping()
            env.env_params["player_name_mappings"] = agent_name_mappings

            # Gather First Obeservation (state)
            state, _ = env.reset()

            # Episode step loop
            for _step in tqdm.tqdm(range(self.TOTAL_EPISODE_STEPS)):
                _action_dictionary = {}

                ################################################
                ################################################
                # Act on the Env (all agents, one after the other)
                ################################################
                for _idx, _agent_name in enumerate(sorted(self.participating_agents.keys())):
                    agent = submission_model if _agent_name == "YOU" else baseline_agents[_agent_name]
                    state_agent = state[_idx]
                    q_values = agent([state_agent])[0]
                    action = q_values.argmax().item()
                    _action_dictionary[_idx] = action

                # Perform action (on all agents)
                state, rewards, done, done, info = env.step(_action_dictionary)

                # Gather rewards for all agents (inside episode_score)
                _step_score = np.array(list(rewards.values()))  # Check with florian about ordering

                episode_scores += _step_score

            # Store the current episode scores
            self.overall_scores.append(episode_scores)
        print("Video directory : ", self.video_directory_path)
        # Post Process Video
        print("Generating Video from thumbnails...")
        video_output_path, video_thumb_output_path = \
            aicrowd_helpers.generate_movie_from_frames(
                self.video_directory_path
            )
        print("Videos : ", video_output_path, video_thumb_output_path)

        # Aggregate all scores into an overall score
        # TODO : Add aggregation function (lets start with simple mean + std)

        self.overall_scores = np.array(self.overall_scores)
        # Compute participant means and stds across episodes
        _score = self.overall_scores.mean(axis=0)
        _score_secondary = self.overall_scores.std(axis=0)

        _idx_of_submitted_agent = self.agent_id("YOU")
        score = _score[_idx_of_submitted_agent]
        score_secondary = _score_secondary[_idx_of_submitted_agent]

        # Post process videos

        print("Scores : ", score, score_secondary)
        print(self.overall_scores)

        _result_object = {
            "score": score,
            "score_secondary": score_secondary,
            "media_video_path": video_output_path,
            "media_video_thumb_path": video_thumb_output_path
        }

        return _result_object


if __name__ == "__main__":
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    answer_file_path = "."
    _client_payload = {}
    _client_payload["submission_file_path"] = "baseline_models/dqn-agent.pt"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = DroneRacerEvaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)

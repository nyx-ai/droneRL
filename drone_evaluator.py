import os.path
import numpy as np
import tqdm
from PIL import Image
import tempfile
import aicrowd_helpers
from torch_impl.agents.dqn import BaseDQNFactory
from torch_impl.env.env import DeliveryDrones
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.helpers.rl_helpers import set_seed
from common.render import Renderer
from torch_impl.render_util import convert_for_rendering


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
        # Evaluation State Variables
        ################################################
        self.EPISODE_SEEDS = [845, 99, 65, 96, 85, 39, 51, 17, 52, 35]
        self.TOTAL_EPISODE_STEPS = 1000
        self.participating_agents = {
            "baseline-1": "sample_models/dqn-agent-1.safetensors",
            "baseline-2": "sample_models/dqn-agent-2.safetensors",
            "baseline-3": "sample_models/dqn-agent-3.safetensors",
            "baseline-4": "sample_models/dqn-agent-4.safetensors",
            "baseline-5": "sample_models/dqn-agent-5.safetensors",
        }

        ################################################
        # Load Baseline models
        ################################################
        self.loaded_agent_models = {}
        for _item in self.participating_agents.keys():
            agent_path = os.path.join(answer_folder_path, self.participating_agents[_item])
            self.loaded_agent_models[_item] = BaseDQNFactory.from_checkpoint(agent_path).create_qnetwork()[0]
        # Baseline Models loaded !! Yayy !!

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
        # Load submission model
        ################################################

        model = BaseDQNFactory.from_checkpoint(submission_file_path).create_qnetwork()[0]
        self.participating_agents["YOU"] = model
        self.loaded_agent_models["YOU"] = model

        self.overall_scores = []

        for _episode_idx, episode_seed in enumerate(self.EPISODE_SEEDS):
            ################################################
            # Run Episode
            ################################################
            episode_scores = np.zeros(len(self.participating_agents.keys()))

            ################################################
            # Env Instantiation
            ################################################
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

            env = WindowedGridView(DeliveryDrones(env_params), radius=3)
            set_seed(env, episode_seed)  # Seed

            agent_name_mappings = self.get_agent_name_mapping()
            env.env_params["player_name_mappings"] = agent_name_mappings

            renderer = Renderer(
                env.n_drones,
                env.side_size,
                resolution_scale_factor=2.0
            )
            renderer.init()

            # Gather First Obeservation (state)
            state = env.reset()

            # Episode step loop
            for _step in tqdm.tqdm(range(self.TOTAL_EPISODE_STEPS)):
                _action_dictionary = {}

                ################################################
                # Act on the Env (all agents, one after the other)
                ################################################
                for _idx, _agent_name in enumerate(sorted(self.participating_agents.keys())):
                    agent = self.loaded_agent_models[_agent_name]

                    ################################################
                    # Gather observation
                    ################################################
                    state_agent = state[_idx]

                    ################################################
                    # Decide action of the participating agent
                    ################################################
                    q_values = agent([state_agent])[0]
                    action = q_values.argmax().item()
                    _action_dictionary[_idx] = action

                # Perform action (on all agents)
                state, rewards, _, _, _ = env.step(_action_dictionary)

                # Gather rewards for all agents (inside episode_score)
                _step_score = np.array(list(rewards.values()))  # Check with florian about ordering

                episode_scores += _step_score

                ################################################
                # Collect frames for the first episode to generate video
                ################################################
                if _episode_idx == 0:
                    if _step < 60:
                        # Use only the first 60 frames for video generation
                        # Record videos with env.render
                        # Do it in a tempfile
                        # Compile frames into a video (from flatland)

                        ground, air, carrying_package, charge = convert_for_rendering(env)
                        _step_frame_im = renderer.render_frame(
                            ground, air, carrying_package, charge, rewards, _action_dictionary)
                        # _step_frame_im = Image.fromarray(np.random.randint(low=0, high=255, size=(250, 250), dtype=np.uint8))
                        _step_frame_im.save("{}/{}.jpg".format(self.video_directory_path, str(_step).zfill(4)))

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
    _client_payload["submission_file_path"] = "sample_models/dqn-agent-1.safetensors"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = DroneRacerEvaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)

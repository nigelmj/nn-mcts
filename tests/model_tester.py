import csv
import re
import warnings
from typing import List

from src.game import Game
from src.model_agent import ModelAgent
from tests.external_mcts import ExternalMCTS
from tests.external_minimax import ExternalMiniMax

warnings.filterwarnings("ignore")


class ModelTester:
    def __init__(
        self,
        game: Game,
        num_simulations: int,
        num_games: int,
        output_csv: str,
        name: str,
    ) -> None:
        self.game = game
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.output_csv = output_csv
        self.name = name

    def set_opponent(self, type: str, name: str, **kwargs) -> None:
        self.opponent_type = type
        self.opponent_name = name
        self.opponent_config = kwargs

    def play_game(self, model_play: bool) -> int:
        opponent_action = -1
        model_action = -1
        if self.opponent_type in ["Model", "Minimax"]:
            for _ in range(1):
                random_action = self.game.get_random_move()
                self.game.make_move(random_action)
                model_play = not model_play

            if model_play:
                opponent_action = random_action
            else:
                model_action = random_action

        while not self.game.is_game_over():
            if model_play:
                model_action = self.model.play_move(
                    self.game, self.num_simulations, opponent_action
                )
                self.game.make_move(model_action)
            else:
                match self.opponent_type:
                    case "MCTS":
                        mcts_agent = ExternalMCTS(
                            self.opponent_config["mcts_sims"], self.game
                        )
                        opponent_action = mcts_agent.get_move(self.game.move_number)
                    case "Minimax":
                        minimax_agent = ExternalMiniMax(
                            self.opponent_config["minimax_depth"], self.game
                        )
                        opponent_action = minimax_agent.get_move()
                    case "Model":
                        opponent_action = self.opp_model.play_move(
                            self.game,
                            self.opponent_config["num_sims"],
                        )
                    case _:
                        opponent_action = self.game.get_random_move()
                self.game.make_move(opponent_action)
            model_play = not model_play
        return self.game.get_winner()

    def run_games(self, iteration_count: int) -> List[List[str]]:
        model_first = []
        opponent_first = []
        for num_game in range(self.num_games // 2):
            self.game.reset()
            self.model.reset(self.game)
            if self.opponent_type == "Model":
                self.opp_model.reset(self.game)

            result = self.play_game(True)
            model_first.append(result)
            print(
                f"Model first, Winner is {result}; Won: {model_first.count(1)}/{num_game + 1}"
            )

            self.game.reset()
            self.model.reset(self.game)
            if self.opponent_type == "Model":
                self.opp_model.reset(self.game)

            result = self.play_game(False)
            opponent_first.append(result)
            print(
                f"Model second, Winner is {result}; Won: {opponent_first.count(-1)}/{num_game + 1}"
            )
            print()

        wins1 = model_first.count(1)
        wins2 = opponent_first.count(-1)
        draws1 = model_first.count(0)
        draws2 = opponent_first.count(0)

        result1 = [str(iteration_count), self.name, self.opponent_name, wins1, draws1]
        result2 = [str(iteration_count), self.opponent_name, self.name, wins2, draws2]
        return [result1, result2]

    def save_results(self, results):
        with open(self.output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            for result in results:
                writer.writerow(result)

    def extract_checkpoint(self, path: str) -> int:
        match = re.search(r"checkpoint_(\d+)", path)
        if not match:
            raise ValueError(f"No checkpoint number found in: {path}")
        return int(match.group(1))

    def test_model(self, model_path: str, **kwargs):
        results = []
        print("Testing model:", model_path)
        self.model = ModelAgent(
            model_path, self.game.size1, self.game.size2, self.game.policy_size
        )
        iteration = self.extract_checkpoint(model_path)

        if self.opponent_type == "Model":
            self.opp_model = ModelAgent(
                self.opponent_config["model_path"],
                self.game.size1,
                self.game.size2,
                self.game.policy_size,
            )

        iteration_results = self.run_games(iteration_count=iteration)
        print("Results", iteration_results)
        results.extend(iteration_results)

        self.save_results(results)

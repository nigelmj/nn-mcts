# from src.alphaZero.inference_server import InferenceServer
from src.games.game import Game
from tensorflow.keras.models import Model
from src.alphaZero.apv_mcts import APVMCTS
from src.alphaZero.apv_node import APVNode
import numpy as np


class Arena:
    def __init__(self,
        game: Game,
        curr_model: Model,
        new_model: Model,
        num_games: int,
        threshold: float,
        num_simulations: int,
        policy_size: int,
        # inference_server: InferenceServer
    ) -> None:
        self.game = game
        self.curr_model = curr_model
        self.new_model = new_model
        self.num_games = num_games
        self.threshold = threshold
        self.num_simulations = num_simulations
        self.policy_size = policy_size
        # self.inference_server = inference_server

    def play_game(self, new_play: bool) -> int:
        while not self.game.is_game_over():
            root = APVNode(self.game.copy(), None, None, 0)
            if new_play:
                mcts = APVMCTS(root, self.new_model, self.num_simulations, self.policy_size, False)
            else:
                mcts = APVMCTS(root, self.curr_model, self.num_simulations, self.policy_size, False)

            improved_policy = mcts.compute_improved_policy()
            action_index = np.argmax(improved_policy)

            row = int(action_index // self.game.size2)
            col = int(action_index % self.game.size2)
            self.game.make_move(row, col)

            new_play = not new_play
        return self.game.get_winner()

    def evaluate_new_model(self) -> int:
        new_play_first = []
        curr_play_first = []
        for _ in range(self.num_games // 2):
            self.game.reset()
            result = self.play_game(True)
            new_play_first.append(result)

        for _ in range(self.num_games // 2):
            self.game.reset()
            result = self.play_game(False)
            curr_play_first.append(result)

        count1 = new_play_first.count(1)
        count2 = curr_play_first.count(-1)
        return count1 + count2

    def cross_threshold(self) -> bool:
        new_wins = self.evaluate_new_model()
        return (new_wins / self.num_games) > self.threshold

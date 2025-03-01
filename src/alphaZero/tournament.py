from src.games.game import Game
from src.alphaZero.pytorch.apv_mcts import APVMCTS
from src.alphaZero.apv_node import APVNode
import numpy as np
import torch


class Tournament:
    def __init__(self,
        game: Game,
        curr_model: torch.nn.Module,
        new_model: torch.nn.Module,
        num_games: int,
        threshold: float,
        num_simulations: int,
    ) -> None:
        self.game = game
        self.curr_model = curr_model
        self.new_model = new_model
        self.num_games = num_games
        self.threshold = threshold
        self.num_simulations = num_simulations

    def play_game(self, new_play: bool) -> int:
        while not self.game.is_game_over():
            root = APVNode(self.game.copy(), None, None, 0)
            if new_play:
                mcts = APVMCTS(root, self.new_model, self.num_simulations, False, False)
            else:
                mcts = APVMCTS(root, self.curr_model, self.num_simulations, False, False)

            improved_policy = mcts.compute_improved_policy()
            action = np.argmax(improved_policy)
            self.game.make_move(action)

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
        print("New Models wins", count1, "games as player 1")
        print("New Model wins", count2, "games as player 2")
        return count1 + count2

    def cross_threshold(self) -> bool:
        new_wins = self.evaluate_new_model()
        return (new_wins / self.num_games) > self.threshold

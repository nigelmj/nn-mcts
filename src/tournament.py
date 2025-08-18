from src.game import Game
from src.mcts import MCTS
from src.node import Node
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
        new_root = Node(self.game.copy(), None, None, 0)
        prev_root = Node(self.game.copy(), None, None, 0)
        while not self.game.is_game_over():
            if new_play:
                mcts = MCTS(new_root, self.new_model, self.num_simulations, False, False)
            else:
                mcts = MCTS(prev_root, self.curr_model, self.num_simulations, False, False)

            improved_policy = mcts.compute_improved_policy()
            action = np.argmax(improved_policy)
            self.game.make_move(action)

            new_play = not new_play

            if action in new_root.children:
                new_root = new_root.children[action]
                new_root.parent = None
            else:
                new_root = Node(self.game.copy(), None, None, 0)

            if action in prev_root.children:
                prev_root = prev_root.children[action]
                prev_root.parent = None
            else:
                prev_root = Node(self.game.copy(), None, None, 0)

        return self.game.get_winner()

    def evaluate_new_model(self) -> int:
        new_play_first = []
        new_play_second = []
        for _ in range(self.num_games // 2):
            self.game.reset()
            result = self.play_game(True)
            new_play_first.append(result)

            self.game.reset()
            result = self.play_game(False)
            new_play_second.append(result)

        count1 = new_play_first.count(1)
        count2 = new_play_second.count(-1)
        draws1 = new_play_first.count(0)
        draws2 = new_play_second.count(0)
        print("New Models wins", count1, "games as player 1")
        print("New Model wins", count2, "games as player 2")
        print("New Model draws", draws1, "games as player 1")
        print("New Model draws", draws2, "games as player 2")
        return count1 + count2

    def cross_threshold(self) -> bool:
        new_wins = self.evaluate_new_model()
        return (new_wins / self.num_games) > self.threshold

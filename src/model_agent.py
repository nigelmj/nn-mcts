import numpy as np
import torch
import torch.multiprocessing as mp

from src.game import Game
from src.mcts import MCTS
from src.neural_network import AlphaZeroNetwork
from src.node import Node


class ModelAgent:
    def __init__(self, model_path: str, size1, size2, policy) -> None:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location="cpu")

        self.model = AlphaZeroNetwork(size1, size2, policy, 2)
        self.model.load_state_dict(state_dict)
        self.model.to("cpu")
        self.model.eval()

    def play_move(
        self,
        game: Game,
        num_sim: int,
        prev_action: int = -1,
    ) -> int:
        if prev_action != -1:
            if prev_action in self.root.children:
                self.root = self.root.children[prev_action]
                self.root.parent = None
            else:
                self.root = Node(game.copy())
        else:
            self.root = Node(game.copy())

        # self.root = Node(game.copy())
        mcts = MCTS(self.root, num_sim, False, False, False, model=self.model)
        policy = mcts.compute_improved_policy()

        action_index = int(np.argmax(policy))
        if action_index in self.root.children:
            self.root = self.root.children[action_index]
            self.root.parent = None

        return action_index

    def reset(self, game: Game):
        self.root = Node(game.copy())

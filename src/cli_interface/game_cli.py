from abc import ABC, abstractmethod
from src.cli_interface.player import PlayerType
from src.mcts.node import Node
from src.mcts.mcts import MonteCarloTreeSearch
from src.games.game import Game

import time


class GameCLI(ABC):
    def __init__(
        self, game: Game, player_pair: tuple[PlayerType, PlayerType], player_1, player_2
    ) -> None:
        self.game = game
        self.player_1_type, self.player_2_type = player_pair
        self.player_1 = player_1
        self.player_2 = player_2

    @abstractmethod
    def display_state(self) -> None:
        raise NotImplementedError

    def get_input(self) -> tuple[int, int]:
        row_end = len(self.game.state)
        col_end = len(self.game.state[0])

        while True:
            try:
                row = int(input(f"Enter row (1-{row_end}): ")) - 1
                col = int(input(f"Enter col (1-{col_end}): ")) - 1
                action = row * self.game.size2 + col

                if not self.game.is_legal_move(action):
                    print("Invalid move. Please choose another.")
                else:
                    return row, col
            except ValueError:
                print("Invalid input. Please enter numbers.")

    def play(self, iterations) -> None:
        while not self.game.is_game_over():
            self.display_state()
            print(
                f"Player {self.player_1 if self.game.current_player == 1 else self.player_2}'s turn"
            )
            if self.game.current_player == 1:
                if self.player_1_type == PlayerType.HUMAN:
                    row, col = self.get_input()
                    action = row * self.game.size2 + col

                elif self.player_1_type == PlayerType.COMPUTER:
                    root = Node(self.game, None, None)
                    mcts = MonteCarloTreeSearch()
                    action = mcts.best_move(root, iterations)
                    row, col = divmod(action, self.game.size2)
                    print(f"Computer plays: {row + 1}, {col + 1}")

                else:
                    action = self.game.get_random_move()

                self.game.make_move(action)

            elif self.game.current_player == -1:
                if self.player_2_type == PlayerType.HUMAN:
                    row, col = self.get_input()
                    action = row * self.game.size2 + col

                elif self.player_2_type == PlayerType.COMPUTER:
                    root = Node(self.game, None, None)
                    mcts = MonteCarloTreeSearch()
                    action = mcts.best_move(root, iterations)
                    row, col = divmod(action, self.game.size2)
                    print(f"Computer plays: {row + 1}, {col + 1}")

                else:
                    action = self.game.get_random_move()

                self.game.make_move(action)

            winner = self.game.get_winner()
            if winner:
                self.display_state()
                print(f"Player {self.player_1 if winner == 1 else self.player_2} wins!")
                return
            # time.sleep(1)
        self.display_state()
        print("It's a draw!")

from abc import ABC, abstractmethod
from src.player import PlayerType
from src.game import Game
from src.tests.model_agent import ModelAgent


class GameCLI(ABC):
    def __init__(
        self, game: Game, player_pair: tuple[PlayerType, PlayerType], player_1, player_2
    ) -> None:
        self.game = game
        self.player_1_type, self.player_2_type = player_pair
        self.player_1 = player_1
        self.player_2 = player_2
        self.model = None

    @abstractmethod
    def display_state(self) -> None:
        raise NotImplementedError

    def initialise_model(self, model_path: str, simulations: int) -> None:
        self.model = ModelAgent(model_path, self.game.size1, self.game.size2, self.game.policy_size)
        self.model.reset(self.game)
        self.simulations = simulations

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

    def play(self) -> None:
        action = None
        play_first = True
        while not self.game.is_game_over():
            self.display_state()
            print("Current Player is:", self.game.current_player)
            print(
                f"Player {self.player_1 if self.game.current_player == 1 else self.player_2}'s turn"
            )
            if play_first:
                if self.player_1_type == PlayerType.HUMAN:
                    row, col = self.get_input()
                    action = row * self.game.size2 + col

                elif self.player_1_type == PlayerType.MODEL:
                    if not self.model:
                        raise ValueError("Model not initialized.")
                    action = self.model.play_move(self.game, self.simulations, action)
                    row, col = divmod(action, self.game.size2)
                    print(f"Model plays: {row + 1}, {col + 1}")

                else:
                    action = self.game.get_random_move()

                self.game.make_move(action)

            else:
                if self.player_2_type == PlayerType.HUMAN:
                    row, col = self.get_input()
                    action = row * self.game.size2 + col

                elif self.player_2_type == PlayerType.MODEL:
                    if not self.model:
                        raise ValueError("Model not initialized.")
                    action = self.model.play_move(self.game, self.simulations, action)
                    row, col = divmod(action, self.game.size2)
                    print(f"Model plays: {row + 1}, {col + 1}")

                else:
                    action = self.game.get_random_move()

                self.game.make_move(action)
            play_first = not play_first

        winner = self.game.get_winner()
        if winner:
            self.display_state()
            print("Winner is:" , winner)
            print(f"Player {self.player_1 if winner == 1 else self.player_2} wins!")
            return
        self.display_state()
        print("It's a draw!")

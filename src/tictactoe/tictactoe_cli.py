from src.cli_interface import GameCLI
from src.player import PlayerType
from src.tictactoe.tictactoe_logic import TicTacToe


class TicTacToeCLI(GameCLI):
    def __init__(self, player_pair: tuple[PlayerType, PlayerType]) -> None:
        super().__init__(TicTacToe(), player_pair, "X", "O")

    def display_state(self) -> None:
        print("\nstate:")
        for row in self.game.state:
            print(" | ".join(["X" if x == 1 else "O" if x == -1 else " " for x in row]))
            print("-" * 9)


if __name__ == "__main__":
    cli = TicTacToeCLI(PlayerType.get_type_pair())
    cli.initialise_model("src/tictactoe/models/TicTacToe_checkpoint_800.pt", 100)
    cli.play()

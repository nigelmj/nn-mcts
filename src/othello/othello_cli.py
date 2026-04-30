from src.cli_interface import GameCLI
from src.othello.othello_logic import Othello
from src.player import PlayerType


class OthelloCLI(GameCLI):
    def __init__(self, player_pair: tuple[PlayerType, PlayerType]) -> None:
        super().__init__(Othello(), player_pair, "Black", "White")

    def display_state(self) -> None:
        print("\nstate:")
        print("   " + "   ".join([str(x) for x in range(1, 9)]))
        print()
        for ind, row in enumerate(self.game.state):
            print(
                str(ind + 1)
                + "  "
                + " | ".join(["○" if x == 1 else "●" if x == -1 else " " for x in row])
            )
            print("   " + "-" * 30)


if __name__ == "__main__":
    cli = OthelloCLI(PlayerType.get_type_pair())
    cli.initialise_model("src/othello/models/othello_checkpoint_300.pt", 50)
    cli.play()

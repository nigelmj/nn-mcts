from src.cli_interface import GameCLI
from src.connect_four.connect_four_logic import ConnectFour
from src.player import PlayerType


class ConnectFourCLI(GameCLI):
    def __init__(self, player_pair: tuple[PlayerType, PlayerType]) -> None:
        super().__init__(ConnectFour(), player_pair, "Red", "Yellow")

    def display_state(self) -> None:
        print("\nstate:")
        print("   " + "   ".join([str(x) for x in range(1, 8)]))
        print()
        for ind, row in enumerate(self.game.state):
            print(
                str(ind + 1)
                + "  "
                + " | ".join(["○" if x == 1 else "●" if x == -1 else " " for x in row])
            )
            print("   " + "-" * 25)


if __name__ == "__main__":
    cli = ConnectFourCLI(PlayerType.get_type_pair())
    cli.initialise_model("src/connect_four/models/connect_four_checkpoint_300.pt", 50)
    cli.play()

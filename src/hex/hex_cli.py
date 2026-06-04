from src.cli_interface import GameCLI
from src.hex.hex_logic import Hex
from src.player import PlayerType


class HexCLI(GameCLI):
    def __init__(self, player_pair: tuple[PlayerType, PlayerType]) -> None:
        super().__init__(Hex(), player_pair, "Red", "Blue")

    def display_state(self) -> None:
        print("\n● goes top to bottom")
        print("○ goes left to right")
        print("state:")
        count = 0
        for row in self.game.state:
            print(
                " " * count
                + " ".join(["●" if x == 1 else "○" if x == -1 else "." for x in row])
            )
            count += 1


if __name__ == "__main__":
    cli = HexCLI(PlayerType.get_type_pair())
    cli.initialise_model("src/hex/models/Hex_checkpoint_30.pt", 500)
    cli.play()

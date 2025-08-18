from src.cli_interface import GameCLI
from src.player import PlayerType
from src.hex.hex_logic import Hex


class HexCLI(GameCLI):
    def __init__(self, player_pair: tuple[PlayerType, PlayerType]) -> None:
        super().__init__(Hex(), player_pair, "R", "B")

    def display_state(self) -> None:
        print("\nstate:")
        count = 0
        for row in self.game.state:
            print(" " * count + " ".join(["R" if x == 1 else "B" if x == -1 else "." for x in row]))
            count += 1

if __name__ == "__main__":
    cli = HexCLI(PlayerType.get_type_pair())
    # cli.initialise_model("src/alphaZero/hex/pt_models/hex_checkpoint_20000.pt", 1000)
    cli.play()

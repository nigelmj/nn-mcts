from src.hex.hex_logic import Hex
from tests.model_tester import ModelTester

if __name__ == "__main__":
    game = Hex()

    model_tester = ModelTester(
        game,
        num_simulations=200,
        num_games=50,
        output_csv="tests/hex/model_vs_mcts.csv",
        name="iter_30",
    )
    model_tester.set_opponent(
        "MCTS",
        "mcts",
        mcts_sims=20000,
    )
    model_tester.test_model("src/hex/models/hex_checkpoint_30.pt")

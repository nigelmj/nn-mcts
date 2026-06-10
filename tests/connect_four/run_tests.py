from src.connect_four.connect_four_logic import ConnectFour
from tests.model_tester import ModelTester

if __name__ == "__main__":
    game = ConnectFour()

    model_tester = ModelTester(
        game,
        num_simulations=200,
        num_games=50,
        output_csv="tests/connect_four/model_vs_mcts.csv",
        name="iter_300",
    )
    model_tester.set_opponent("MCTS", mcts_sims=50000, name="mcts")
    model_tester.test_model("src/connect_four/models/connect_four_checkpoint_300.pt")

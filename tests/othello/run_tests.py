from src.othello.othello_logic import Othello
from tests.model_tester import ModelTester

if __name__ == "__main__":
    game = Othello()

    model_tester = ModelTester(
        game,
        num_simulations=200,
        num_games=50,
        output_csv="tests/othello/model_vs_mcts.csv",
        name="iter_300",
    )
    model_tester.set_opponent("MCTS", mcts_sims=50000, name="mcts")
    model_tester.test_model("src/othello/models/othello_checkpoint_300.pt")

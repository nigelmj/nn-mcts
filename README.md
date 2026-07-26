# Neural Network aided MCTS

AlphaZero-style self-play reinforcement learning for board games using Monte Carlo Tree Search and a dual-head policy-value neural network.

**Games implemented:** Tic-Tac-Toe (3×3), Connect Four (6×7), Othello (8×8), Hex (11×11 / 7×7).

## Setup

```bash
pip install -r requirements.txt
```

Requires numpy, torch, and scipy (Connect Four only).

## Training

```bash
python -m src.tictactoe.tictactoe_train
python -m src.connect_four.connect_four_train
python -m src.othello.othello_train
python -m src.hex.hex_train
```

Each script runs iterative self-play + training with configurable hyperparameters.

A note on games per iteration: 7×7 Hex performs well with 100 games per iteration, but 11×11 Hex needs ~1000 games to generalise well after training. Larger state spaces need more games per iteration — otherwise the model overfits to subpar moves due to insufficient exploration.

## Play

```bash
python -m src.tictactoe.tictactoe_cli
python -m src.connect_four.connect_four_cli
python -m src.othello.othello_cli
python -m src.hex.hex_cli
```

Play as human, against a trained model, or against a random agent.

## Parallelised Workflow

Training spawns one worker process per CPU core (minus one) to generate self-play games in parallel. All workers share a dedicated inference process (`InferenceWorker`) that batches neural network evaluations, enabling efficient GPU utilisation and avoiding GIL contention.

## Adding a New Game

Subclass `Game` (in `src/game.py`) and implement:

- `make_move`, `get_legal_moves`, `get_winner`, `is_game_over`
- `encode_state` → 2-channel numpy array (channel 0 = current player, channel 1 = opponent). Two things must hold:
  1. **Current-player perspective** — the network always sees the board from the side to move, so it learns one policy/value function, not two.
  2. **Preserved goal direction** — if the game has asymmetric goals (e.g. Hex: connect top-to-bottom vs left-to-right), the encoding must keep the same relative goal regardless of which player is to move. In Hex this means rotating/flipping so the network always sees "top-to-bottom", then transforming the policy indices back after inference. This was an easy bug to miss.
- `legal_moves_mask` → boolean array with valid moves and action count
- `mask_normalise_policy` → apply mask and renormalise

Then subclass `GameZero` (in `src/train_pipeline.py`) to define `augment_data` with the appropriate symmetries and a `training_config` dictionary. When implementing `augment_data`, ensure any symmetry transforms also preserve goal direction and map policy indices back to the original board's coordinate system.

## Testing

Test trained models against minimax and plain MCTS baselines (requires Java agents at `tests/java/agents/`):

```bash
python -m tests.connect_four.run_tests
python -m tests.othello.run_tests
python -m tests.hex.run_tests
```

Subtree reuse (keeping the MCTS tree across moves instead of rebuilding from scratch) performs significantly better in both training and testing. This was consistent across all three larger games tested.

## Testing Results

*To be filled — results against baseline MCTS, minimax, and the impact of subtree reuse need to be re-run.*

## Architecture

- **`src/game.py`** — Abstract game interface
- **`src/neural_network.py`** — ResNet-style CNN with policy & value heads
- **`src/node.py`** / **`src/mcts.py`** — PUCT-based MCTS
- **`src/train_pipeline.py`** — Self-play loop, replay buffer, training
- **`src/inference_worker.py`** — Separate process for batched NN inference
- **`src/model_agent.py`** — Loads a checkpoint and plays via MCTS

## License

MIT

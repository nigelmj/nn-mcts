from abc import ABC, abstractmethod
from src.games.game import Game
from src.alphaZero.apv_node import APVNode
from src.alphaZero.apv_mcts import APVMCTS
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
from typing import List, Tuple, Dict


print('TensorFlow version:',tf.__version__)
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print("CUDA version:",cuda_version)
cudnn_version = sys_details["cudnn_version"]
print("CUDNN version:",cudnn_version)


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample_all(self):
        states, policies, values = zip(*self.buffer)
        return np.array(states), np.array(policies), np.array(values)

    def __len__(self):
        return len(self.buffer)


class GameZero(ABC):
    def __init__(self, game: Game, policy_size: int) -> None:
        self.model = None
        self.game = game
        self.policy_size = policy_size

    def build_network(self, num_channels: int) -> Model:
        input_shape = (num_channels, self.game.size1, self.game.size2)
        inputs = layers.Input(shape=input_shape, dtype="float16")

        tf.keras.backend.set_image_data_format("channels_first")

        x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(
            inputs
        )

        for _ in range(5):
            skip = x
            x = layers.Conv2D(64, kernel_size=(3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(64, kernel_size=(3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, skip])
            x = layers.ReLU()(x)

        x = layers.Flatten()(x)

        x = layers.Flatten()(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        policy_output = layers.Dense(
            self.policy_size,
            activation="softmax",
            name="policy",
            dtype="float32",
        )(x)
        value_output = layers.Dense(
            1, activation="tanh", name="value", dtype="float32"
        )(x)

        self.model = models.Model(inputs=inputs, outputs=[policy_output, value_output])

        # optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        self.model.compile(
            optimizer="adam",
            loss={"policy": "categorical_crossentropy", "value": "mean_squared_error"},
            loss_weights={"policy": 1.0, "value": 1.0},
        )

        return self.model

    def generate_games(
        self, num_simulations: int, stochastic_threshold: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.game.reset()
        states, policies, values = [], [], []

        while not self.game.is_game_over():
            state = self.game.encode_state()

            root = APVNode(self.game.copy(), None, None, 0)
            mcts = APVMCTS(root, self.model, num_simulations, 122, True)
            improved_policy = mcts.compute_improved_policy()

            states.append(state)
            policies.append(improved_policy)

            row, col = self.get_move(len(states), stochastic_threshold, improved_policy)
            self.game.make_move(row, col)

        result = self.game.get_winner()

        # Values are flipped as they are from the perspective of the current player
        values = [result if i % 2 == 0 else -result for i in range(len(states))]

        return self.augment_data(states, policies, values)

    @abstractmethod
    def get_move(self, states_len: int, stochastic_threshold: int, improved_policy: np.ndarray) -> Tuple[int, int]:
        pass

    @abstractmethod
    def augment_data(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def train_network(self, replay_buffer, num_epochs: int, batch_size: int) -> None:
        if len(replay_buffer) < batch_size:
            return

        # TODO: Update sample to prefer more recent data
        states, policies, values = replay_buffer.sample_all()

        history = self.model.fit(
            x=states,
            y={"policy": policies, "value": values},
            batch_size=batch_size,
            epochs=num_epochs,
        )
        return history

    def training_pipeline(self, config: Dict) -> None:
        replay_buffer = ReplayBuffer(config["replay_buffer_size"])
        games_played = 0

        for _ in range(config["iterations"]):
            # Generate self-play games
            num_games_per_iteration = config["games_per_iteration"]
            for _ in range(num_games_per_iteration):
                states, policies, values = self.generate_games(
                    config["num_simulations"],
                    config["stochastic_threshold"]
                )
                # Add game data to replay buffer
                for state, policy, value in zip(states, policies, values):
                    replay_buffer.add(state, policy, value)
                games_played += 1
                print(f"Completed game {games_played}")

            # Train on many batches from replay buffer
            self.train_network(
                replay_buffer,
                num_epochs=config["num_epochs"],
                batch_size=config["batch_size"],
            )

            # Save checkpoint
            if games_played % config["checkpoint_frequency"] == 0:
                self.model.save(f"models/hex_checkpoint_{games_played}.keras")

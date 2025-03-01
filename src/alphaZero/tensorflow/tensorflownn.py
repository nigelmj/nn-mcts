from abc import ABC, abstractmethod
from src.games.game import Game
from src.alphaZero.apv_node import APVNode
from src.alphaZero.tensorflow.apv_mcts import APVMCTS
from src.alphaZero.tournament import Tournament
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
from random import shuffle
from typing import List, Tuple, Dict


print('TensorFlow version:',tf.__version__)
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print("CUDA version:",cuda_version)
cudnn_version = sys_details["cudnn_version"]
print("CUDNN version:",cudnn_version)


class GameZero(ABC):
    def __init__(self, game: Game) -> None:
        self.model = None
        self.game = game

    def build_network(self, num_channels: int) -> Model:
        input_shape = (num_channels, self.game.size1, self.game.size2)
        inputs = layers.Input(shape=input_shape, dtype="float64")

        tf.keras.backend.set_image_data_format("channels_first")

        x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.ReLU()(x)

        for _ in range(5):
            skip = x
            x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization(axis=1)(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization(axis=1)(x)
            x = layers.Add()([x, skip])
            x = layers.ReLU()(x)
        
        policy_output = layers.Conv2D(2, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        policy_output = layers.BatchNormalization(axis=1)(policy_output)
        policy_output = layers.ReLU()(policy_output)
        policy_output = layers.Flatten()(policy_output)
        policy_output = layers.Dense(
            self.game.policy_size,
            activation="softmax",
            name="policy",
            dtype="float64",
        )(policy_output)

        value_output = layers.Conv2D(1, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        value_output = layers.BatchNormalization(axis=1)(value_output)
        value_output = layers.ReLU()(value_output)
        value_output = layers.Flatten()(value_output)
        value_output = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(value_output)
        value_output = layers.ReLU()(value_output)
        value_output = layers.Dense(
            1, activation="tanh", name="value", dtype="float64"
        )(value_output)

        self.model = models.Model(inputs=inputs, outputs=[policy_output, value_output])

        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        self.model.compile(
            optimizer=optimizer,
            loss={"policy": "categorical_crossentropy", "value": "mean_squared_error"},
            loss_weights={"policy": 1.0, "value": 1.0},
        )

        return self.model

    def generate_games(
        self, num_simulations: int, sampling: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.game.reset()
        states, policies, values = [], [], []

        while not self.game.is_game_over():
            state = self.game.encode_state()

            root = APVNode(self.game.copy(), None, None, 0)
            mcts = APVMCTS(root, self.model, num_simulations, True, sampling)
            improved_policy = mcts.compute_improved_policy()

            states.append(state)
            policies.append(improved_policy)

            action = np.random.choice(len(improved_policy), p=improved_policy)
            self.game.make_move(action)

        result = self.game.get_winner()

        # Values are flipped as they are from the perspective of the current player
        values = [result if i % 2 == 0 else -result for i in range(len(states))]

        return self.augment_data(states, policies, values)

    @abstractmethod
    def augment_data(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def train_network(self, training_samples, num_epochs: int, batch_size: int) -> None:

        states, policies, values = zip(*training_samples)
        states, policies, values = np.array(states), np.array(policies), np.array(values)

        history = self.model.fit(
            x=states,
            y={"policy": policies, "value": values},
            batch_size=batch_size,
            epochs=num_epochs,
        )
        return history

    def training_pipeline(self, config: Dict) -> None:
        games_played = 0
        training_data = []

        for _ in range(config["iterations"]):

            # Generate self-play games
            num_games_per_iteration = config["games_per_iteration"]
            episode_data = deque(maxlen=config['episode_data_size'])
            for _ in range(num_games_per_iteration):
                states, policies, values = self.generate_games(
                    config["num_simulations"],
                    config["stochastic_threshold"] > (games_played % config["games_per_iteration"])
                )

                for state, policy, value in zip(states, policies, values):
                    episode_data.append((state, policy, value))
                games_played += 1

            print("Games played: ", games_played)
            training_data.append(episode_data)

            if len(training_data) > config['max_iter_per_train_step']:
                print("Removing old data")
                training_data.pop(0)

            training_samples = []
            for e in training_data:
                training_samples.extend(e)
            shuffle(training_samples)

            old_model = tf.keras.models.clone_model(self.model)
            old_model.set_weights(self.model.get_weights())
            
            self.train_network(
                training_samples,
                num_epochs=config["num_epochs"],
                batch_size=config["batch_size"],
            )

            tournament = Tournament(
                game=self.game,
                curr_model=old_model,
                new_model=self.model,
                num_games=config["tournament_games"],
                threshold=config["update_threshold"],
                num_simulations=config["num_simulations"],
            )

            if not tournament.cross_threshold():
                print("New model rejected; reverting to the old model.")
                self.model.set_weights(old_model.get_weights())
            else:
                print("New model accepted based on tournament evaluation.")
            # Save checkpoint
            if games_played % config["checkpoint_frequency"] == 0:
                self.model.save(f"{config['path']}_checkpoint_{games_played}")
                self.model.save(f"{config['keras_path']}_checkpoint_{games_played}.keras")

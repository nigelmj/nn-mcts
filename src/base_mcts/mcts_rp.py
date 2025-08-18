from concurrent.futures import ProcessPoolExecutor
from src.mcts.node import Node
import random


class MCTSRootParallel:
    def __init__(self) -> None:
        pass

    def best_move(self, root: Node, simulations_number: int, num_processes: int = 8) -> tuple[int, int]:

        simulations_per_process = simulations_number // num_processes
        root_copies = [root.copy() for _ in range(num_processes)]

        with ProcessPoolExecutor() as executor:
            results = executor.map(self._run_simulations, root_copies, [simulations_per_process] * num_processes)

        for result_root in results:
            for child in root.get_children():
                for result_child in result_root.get_children():
                    if child.move == result_child.move:
                        child.wins += result_child.wins
                        child.simulations += result_child.simulations

        best_child = max(root.get_children(), key=lambda child: (child.simulations, child.wins))

        if not best_child.move:
            return (-1, -1)

        return best_child.move

    def _run_simulations(self, root_copy: Node, num_simulations: int):
        for _ in range(num_simulations):
            node = self._selection(root_copy)
            expanded_node = self._expansion(node)
            result = self._simulation(expanded_node)
            self._backpropagation(expanded_node, result)
        return root_copy

    def _selection(self, root: Node) -> Node:
        node = root
        while node.is_fully_explored() and not node.is_terminal():
            node = node.best_child()
        return node

    def _expansion(self, selected_node: Node) -> Node:
        if selected_node.is_terminal():
            return selected_node

        children = selected_node.get_children()
        unexplored_children = [child for child in children if child.simulations == 0]
        return random.choice(unexplored_children)

    def _simulation(self, expanded_node: Node) -> int:
        simulated_game = expanded_node.game.copy()
        while not simulated_game.is_game_over():
            move = random.choice(simulated_game.get_legal_moves())
            simulated_game.make_move(*move)
        return simulated_game.get_winner()

    def _backpropagation(self, terminal_node: Node, result: int) -> None:
        node = terminal_node
        while node is not None:
            node.wins += abs(result) if node.game.current_player != result else -abs(result)
            node.simulations += 1
            node = node.parent

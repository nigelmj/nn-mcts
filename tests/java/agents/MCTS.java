package tests.java.agents;

import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;

import tests.java.games.BoardState;
import tests.java.games.Move;

public class MCTS {
    public MCTS() {
    }

    public Move bestMove(Node root, int iterations) {
        for (int i = 0; i < iterations; i++) {
            Node selectedNode = _selection(root);
            Node expandedNode = _expansion(selectedNode);
            int result = _simulation(expandedNode);
            _backpropagation(expandedNode, result);
        }

        Node bestChild = root
            .getChildren()
            .stream()
            .max((child1, child2) ->
                Integer.compare(child1.getVisits(), child2.getVisits())
            )
            .orElseThrow(() -> new RuntimeException("No best child found"));

        return bestChild.getMove();
    }

    private Node _selection(Node root) {
        Node node = root;
        while (node.isFullyExpanded() && !node.isTerminal()) {
            node = node.bestChild(1.41);
        }
        return node;
    }

    private Node _expansion(Node selectedNode) {
        if (selectedNode.isTerminal()) {
            return selectedNode;
        }
        ArrayList<Node> children = selectedNode.getChildren();
        return children
            .stream()
            .filter(child -> child.getVisits() == 0)
            .findAny()
            .orElseThrow(() -> new RuntimeException("No unexplored children"));
    }

    private int _simulation(Node expandedNode) {
        BoardState simulatedGame = expandedNode.getState().copy();
        while (!simulatedGame.isGameOver()) {
            Move move = randomMove(simulatedGame);
            simulatedGame.makeMove(move);
        }
        return simulatedGame.getWinner();
    }

    private void _backpropagation(Node terminalNode, int result) {
        Node node = terminalNode;
        while (node != null) {
            int value = result != node.getState().getPlayer() ? Math.abs(result) : -Math.abs(result);
            node.setValue(node.getValue() + value);
            node.setVisits(node.getVisits() + 1);
            node = node.getParent();
        }
    }

    private Move randomMove(BoardState simulatedGame) {
        ArrayList<Move> legalMoves = simulatedGame.getLegalMoves();
        Random rand = new Random();
        return legalMoves.get(rand.nextInt(legalMoves.size()));
    }
}
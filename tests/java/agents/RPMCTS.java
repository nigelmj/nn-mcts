package tests.java.agents;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.lang.Math;
import java.util.concurrent.*;

import tests.java.games.BoardState;
import tests.java.games.Move;

public class RPMCTS {
    public RPMCTS() {
    }

    public Move bestMove(Node root, int iterations) {
        final int numThreads = 4;

        Node[] rootCopies = new Node[numThreads];
        for (int i = 0; i < numThreads; i++) {
            rootCopies[i] = new Node(root); 
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Node>> futures = new ArrayList<>();

        for (Node rootCopy : rootCopies) {
            futures.add(
                executor.submit(() ->
                    runSimulations(rootCopy, (int) iterations / numThreads)
                )
            );
        }

        try {
            for (Future<Node> future : futures) {
                Node resultRoot = future.get();
                for (Node child : root.getChildren()) {
                    for (Node resultChild : resultRoot.getChildren()) {
                        if (child.getMove().equals(resultChild.getMove())) {
                            child.incrementValue(resultChild.getValue());
                            child.incrementVisits(resultChild.getVisits());
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
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

    private Node runSimulations(Node rootCopy, int numSimulations) {
        for (int i = 0; i < numSimulations; i++) {
            Node selectedNode = _selection(rootCopy);
            Node expandedNode = _expansion(selectedNode);
            int result = _simulation(expandedNode);
            _backpropagation(expandedNode, result);
        }
        return rootCopy;
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
package tests.java.agents;

import java.util.ArrayList;
import tests.java.games.BoardState;
import tests.java.games.Move;

public class Node {

    private BoardState state;
    private Node parent;
    private Move move;
    private ArrayList<Node> children;
    private double value;
    private int visits;

    public Node(BoardState state, Node parent, Move move) {
        this.state = state;
        this.parent = parent;
        this.move = move;
        this.children = null;
        this.value = 0;
        this.visits = 0;
    }

    public Node(Node copyNode) {
        this.state = copyNode.state.copy();
        this.parent = copyNode.parent;
        this.move = copyNode.move;
        this.children = null;
        this.value = copyNode.value;
        this.visits = copyNode.visits;
    }

    public void incrementVisits(int count) {
        this.visits += count;
    }

    // Increment wins
    public void incrementValue(double value) {
        this.value += value;
    }

    public boolean isFullyExpanded() {
        return (
            children != null &&
            children.stream().allMatch(child -> child.visits > 0)
        );
    }

    public boolean isTerminal() {
        return state.isGameOver();
    }

    private double uct(double c) {
        if (visits == 0 || parent == null) {
            return Double.POSITIVE_INFINITY;
        }
        double exploitation = value / visits;
        double exploration = c * Math.sqrt(Math.log(parent.visits) / visits);
        return exploitation + exploration;
    }

    public Node bestChild(double c) {
        if (children == null || children.isEmpty()) {
            return this; // No children; return self
        }
        return children
            .stream()
            .max((child1, child2) ->
                Double.compare(child1.uct(c), child2.uct(c))
            )
            .orElse(this);
    }

    public ArrayList<Node> getChildren() {
        if (children == null) {
            children = new ArrayList<>();
            for (Move legalMove : state.getLegalMoves()) {
                BoardState newState = state.copy();
                newState.makeMove(legalMove);
                children.add(new Node(newState, this, legalMove));
            }
        }
        return children;
    }

    public BoardState getState() {
        return state;
    }

    public Node getParent() {
        return parent;
    }

    public Move getMove() {
        return move;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public int getVisits() {
        return visits;
    }

    public void setVisits(int visits) {
        this.visits = visits;
    }
}

package tests.java.games;

import java.util.ArrayList;
import java.util.HashSet;

public class HexState extends BoardState {

    private int[][] state;
    private int player;
    private int turn;

    public HexState(int[][] state, int player, int turn) {
        this.state = state;
        this.player = player;
        this.turn = turn;
    }

    public HexState(HexState parent) {
        this.state = new int[parent.state.length][parent.state[0].length];
        for (int i = 0; i < parent.state.length; i++) {
            for (int j = 0; j < parent.state[i].length; j++) {
                this.state[i][j] = parent.state[i][j];
            }
        }
        this.player = parent.player;
        this.turn = parent.turn;
    }

    @Override
    public int getPlayer() {
        return this.player;
    }

    @Override
    public int[][] getState() {
        return this.state;
    }

    @Override
    public ArrayList<Move> getLegalMoves() {
        ArrayList<Move> legalMoves = new ArrayList<Move>();
        for (int i = 0; i < state.length; i++) {
            for (int j = 0; j < state[i].length; j++) {
                if (state[i][j] == 0) {
                    legalMoves.add(new Move(i, j));
                }
            }
        }
        if (turn == 2) {
            legalMoves.add(new Move(-1, -1));
            return legalMoves;
        }
        return legalMoves;
    }

    @Override
    public boolean isLegalMove(Move move) {
        if (this.turn == 2 && move.getX() == -1 && move.getY() == -1) {
            return true;
        }
        return state[move.getX()][move.getY()] == 0;
    }

    @Override
    public void makeMove(Move move) {
        if (turn == 2 && move.getX() == -1 && move.getY() == -1) {
            player = player * -1;
        } else {
            state[move.getX()][move.getY()] = player;
        }
        player = player * -1;
        turn++;
    }

    @Override
    public boolean isGameOver() {
        if (turn < 21) {
            return false;
        }
        return getWinner()!=0;
    }

    @Override
    public int getWinner() {
        int n = state.length; // Board size
        HashSet<int[]> visited = new HashSet<>();
        player *= -1;

        // Determine the edges to check based on the player
        HashSet<int[]> startPositions = new HashSet<>();
        GoalChecker goalChecker;

        if (player == 1) { // Red connects top to bottom
            for (int col = 0; col < n; col++) {
                if (state[0][col] == player) {
                    startPositions.add(new int[] { 0, col });
                }
            }
            goalChecker = (x, y) -> x == n - 1; // Reached bottom row
        } else { // Blue connects left to right
            for (int row = 0; row < n; row++) {
                if (state[row][0] == player) {
                    startPositions.add(new int[] { row, 0 });
                }
            }
            goalChecker = (x, y) -> y == n - 1; // Reached right column
        }

        // Directions for Hex adjacency
        int[][] directions = {
            { -1, 0 },
            { 1, 0 },
            { 0, -1 },
            { 0, 1 },
            { -1, 1 },
            { 1, -1 },
        };

        // Start DFS from all valid starting positions
        for (int[] pos : startPositions) {
            if (
                dfs(pos[0], pos[1], visited, goalChecker, directions, n, player)
            ) {
                player *= -1;
                return this.state[pos[0]][pos[1]];
            }
        }

        player *= -1;
        return 0; // No winner
    }

    // Depth First Search (DFS)
    boolean dfs(
        int x,
        int y,
        HashSet<int[]> visited,
        GoalChecker goalChecker,
        int[][] directions,
        int n,
        int player
    ) {
        if (goalChecker.checkGoal(x, y)) {
            return true;
        }

        visited.add(new int[] { x, y });
        for (int[] dir : directions) {
            int nx = x + dir[0], ny = y + dir[1];
            if (
                nx >= 0 &&
                nx < n &&
                ny >= 0 &&
                ny < n &&
                !contains(visited, nx, ny)
            ) {
                if (
                    state[nx][ny] == player &&
                    dfs(nx, ny, visited, goalChecker, directions, n, player)
                ) {
                    return true;
                }
            }
        }
        return false;
    }

    // Helper method to check if a HashSet<int[]> contains a specific coordinate
    private boolean contains(HashSet<int[]> set, int x, int y) {
        for (int[] coord : set) {
            if (coord[0] == x && coord[1] == y) {
                return true;
            }
        }
        return false;
    }

    @FunctionalInterface
    private interface GoalChecker {
        boolean checkGoal(int x, int y);
    }

    int getTurn() {
        return turn;
    }

    @Override
    public BoardState copy() {
        return new HexState(this);
    }

    @Override
    public int getScore() {
        int score = 0;
        for (int i = 0; i < state.length; i++) {
            for (int j = 0; j < state[i].length; j++) {
                if (state[i][j] == 1) {
                    score++;
                } else if (state[i][j] == -1) {
                    score--;
                }
            }
        }
        return score;
    }

    @Override
    public int boardEval() {
        return getScore();
    }

    public void display() {
        System.out.println("  0 1 2 3 4 5 6 7 8 9 10");
        for (int i = 0; i < state.length; i++) {
            System.out.print(i + " ".repeat(i) +" ");
            for (int j = 0; j < state[i].length; j++) {
                char symbol;
                if (state[i][j] == 1) {
                    symbol = 'R';
                } else if (state[i][j] == -1) {
                    symbol = 'B';
                } else {
                    symbol = '.';
                }
                System.out.print(symbol + " ");
            }
            System.out.println();
        }
    }
}

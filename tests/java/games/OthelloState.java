package tests.java.games;

import java.util.ArrayList;

public class OthelloState extends BoardState {
    private int[][] state;
    private int player;
    private ArrayList<Move> legalMoves;

    public OthelloState(int[][] state, int player) {
        this.state = state;
        this.player = player;
        this.legalMoves = null;
    }

    public OthelloState(OthelloState parent) {
        this.state = new int[parent.state.length][parent.state[0].length];
        for (int i = 0; i < parent.state.length; i++) {
            for (int j = 0; j < parent.state[i].length; j++) {
                this.state[i][j] = parent.state[i][j];
            }
        }
        this.player = parent.player;
        if (parent.legalMoves != null) {
            this.legalMoves = new ArrayList<Move>();
            for (Move move : parent.legalMoves) {
                this.legalMoves.add(new Move(move.getX(), move.getY()));
            }
        } else {
            this.legalMoves = getLegalMoves();
        }
    }

    @Override
    public BoardState copy() {
        return new OthelloState(this);
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
        ArrayList<Move> possibleMoves = getPossibleMoves();
        if (possibleMoves.isEmpty()) {
            ArrayList<Move> skipMove = new ArrayList<>();
            skipMove.add(new Move(-1, -1));
            return skipMove;
        }
        return possibleMoves;
    }

    private ArrayList<Move> getPossibleMoves() {
        if (legalMoves != null) {
            return legalMoves;
        }
        legalMoves = new ArrayList<>();
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (isLegalMove(row, col)) {
                    legalMoves.add(new Move(row, col));
                }
            }
        }
        return legalMoves;
    }

    @Override
    public boolean isLegalMove(Move move) {
        // The "skip" move is always legal.
        if (move.getX() == -1 && move.getY() == -1) {
            return true;
        }
        return isLegalMove(move.getX(), move.getY());
    }

    public boolean isLegalMove(int row, int col) {
        if (!inBounds(row, col) || state[row][col] != 0) {
            return false;
        }
        for (int dRow = -1; dRow <= 1; dRow++) {
            for (int dCol = -1; dCol <= 1; dCol++) {
                if (dRow == 0 && dCol == 0) continue;
                if (isLegalDirection(row, col, dRow, dCol)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean isLegalDirection(int row, int col, int dRow, int dCol) {
        int newRow = row + dRow;
        int newCol = col + dCol;
        if (!inBounds(newRow, newCol) || state[newRow][newCol] != -player) {
            return false;
        }
        newRow += dRow;
        newCol += dCol;
        while (inBounds(newRow, newCol) && state[newRow][newCol] == -player) {
            newRow += dRow;
            newCol += dCol;
        }
        return inBounds(newRow, newCol) && state[newRow][newCol] == player;
    }

    private boolean inBounds(int row, int col) {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }

    @Override
    public void makeMove(Move move) {
        // Invalidate the legal moves cache.
        legalMoves = null;
        // If it's a "skip" move, simply switch the player.
        if (move.getX() == -1 && move.getY() == -1) {
            player = -player;
            return;
        }
        state[move.getX()][move.getY()] = player;
        flipPieces(move.getX(), move.getY());
        player = -player;
    }

    private void flipPieces(int row, int col) {
        for (int dRow = -1; dRow <= 1; dRow++) {
            for (int dCol = -1; dCol <= 1; dCol++) {
                if (dRow == 0 && dCol == 0) continue;
                flipDirection(row, col, dRow, dCol);
            }
        }
    }

    private void flipDirection(int row, int col, int dRow, int dCol) {
        int newRow = row + dRow;
        int newCol = col + dCol;
        while (inBounds(newRow, newCol) && state[newRow][newCol] == -player) {
            newRow += dRow;
            newCol += dCol;
        }
        if (inBounds(newRow, newCol) && state[newRow][newCol] == player) {
            int flipRow = row + dRow;
            int flipCol = col + dCol;
            while (flipRow != newRow || flipCol != newCol) {
                state[flipRow][flipCol] = player;
                flipRow += dRow;
                flipCol += dCol;
            }
        }
    }

    @Override
    public int getWinner() {
        int sum = 0;
        for (int i= 0; i < 8; i++) {
            for (int j=0; j < 8; j++) {
                sum = sum + state[i][j];
            }
        }
        if (sum>0) return 1;
        else if (sum<0) return -1;
        return 0;
    }

    @Override
    public boolean isGameOver() {
        ArrayList<Move> moves = getPossibleMoves();
        if (moves.isEmpty()) {
            player = -player;
            ArrayList<Move> opponentMoves = getPossibleMoves();
            player = -player;
            if (opponentMoves.isEmpty()) {
                return true;
            }
        }
        return false; 
    }

    public void display() {
        // Print a header with column indices
        System.out.println("  0 1 2 3 4 5 6 7");
        for (int i = 0; i < state.length; i++) {
            System.out.print(i + " ");
            for (int j = 0; j < state[i].length; j++) {
                char symbol;
                if (state[i][j] == 1) {
                    symbol = 'B';
                } else if (state[i][j] == -1) {
                    symbol = 'W';
                } else {
                    symbol = '.';
                }
                System.out.print(symbol + " ");
            }
            System.out.println();
        }
    }

    public int getScore() {
        int score = 0;
        for (int i = 0; i < state.length; i++) {
            for (int j = 0; j < state[i].length; j++) {
                score += state[i][j];
            }
        }
        return score;
    }

    public int boardEval() {
        int ans= 0;
        for (int i=0; i<8; i++)
            for (int j=0; j<8; j++)
                if (state[i][j] == 1) {
                    ans+= getBoardPosition()[i][j];
                } else if (state[i][j] == -1) {
                    ans-= getBoardPosition()[i][j];
                }
        return ans;
    }

    public int[][] getBoardPosition() {
        return new int[][] {
            {120, -20, 20, 5, 5, 20, -20, 120},
            {-20, -40, -5, -5, -5, -5, -40, -20},
            {20, -5, 15, 3, 3, 15, -5, 20},
            {5, -5, 3, 3, 3, 3, -5, 5},
            {5, -5, 3, 3, 3, 3, -5, 5},
            {20, -5, 15, 3, 3, 15, -5, 20},
            {-20, -40, -5, -5, -5, -5, -40, -20},
            {120, -20, 20, 5, 5, 20, -20, 120}
        };
    }
}

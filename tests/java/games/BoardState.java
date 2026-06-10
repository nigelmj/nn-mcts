package tests.java.games;

import java.util.ArrayList;

public abstract class BoardState {
    public abstract int getPlayer();

    public abstract int[][] getState();

    public abstract ArrayList<Move> getLegalMoves();

    public abstract void makeMove(Move move);

    public abstract boolean isGameOver();

    public abstract int getWinner();

    public abstract boolean isLegalMove(Move move);

    public abstract BoardState copy();

    public abstract int getScore();

    public abstract int boardEval();
}

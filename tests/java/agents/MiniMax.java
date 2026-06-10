package tests.java.agents;

import java.util.ArrayList;

import tests.java.games.BoardState;
import tests.java.games.Move;

public class MiniMax {
    public int minimax(BoardState boardState, int depth, int alpha, int beta) {
        if (depth == 0) {
            return boardState.boardEval();
        }

        if (boardState.isGameOver()) {
            if (boardState.getScore() > 0) {
                return Integer.MAX_VALUE-1;
            } else if (boardState.getScore() < 0) {
                return Integer.MIN_VALUE+1;
            } else {
                return 0;
            }
        }

        else if (boardState.getPlayer() == 1) {
            int maxEval = Integer.MIN_VALUE;
            ArrayList<Move> moves = boardState.getLegalMoves();
            for (Move move: moves) {
                BoardState board = boardState.copy();
                board.makeMove(move);

                int eval = minimax(board, depth-1, alpha, beta);
                maxEval = Math.max(maxEval, eval);

                alpha = Math.max(alpha, maxEval);
                if (beta <= alpha) {
                    break;
                }
            }
            return maxEval;
        }

        else {
            int minEval = Integer.MAX_VALUE;
            ArrayList<Move> moves = boardState.getLegalMoves();
            for (Move move: moves) {
                BoardState board = boardState.copy();
                board.makeMove(move);

                int eval = minimax(board, depth-1, alpha, beta);
                minEval = Math.min(minEval, eval);

                beta = Math.min(beta, minEval);
                if (beta <= alpha) {
                    break;
                }
            }
            return minEval;
        }
    }

    public Move bestMove(BoardState boardState, int searchDepth) {
        Move ans = new Move(-1, -1);

        ArrayList<Move> moves = boardState.getLegalMoves();
        int highest = Integer.MIN_VALUE;
        int lowest = Integer.MAX_VALUE;
        int alpha = Integer.MIN_VALUE;
        int beta = Integer.MAX_VALUE;

        if (boardState.getPlayer() == 1){
            for (Move move: moves) {
                BoardState board = boardState.copy();
                board.makeMove(move);

                int eval = minimax(board, searchDepth-1, alpha, beta);
                if (highest<eval) {
                    highest = eval;
                    ans = move;
                }
            }

        } else {
            for (Move move: moves) {
                BoardState board = boardState.copy();
                board.makeMove(move);

                int eval = minimax(board, searchDepth-1, alpha, beta);
                if (lowest>eval) {
                    lowest = eval;
                    ans = move;
                }
            }
        }
        return ans;
    }
}

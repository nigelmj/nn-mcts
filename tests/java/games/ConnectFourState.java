package tests.java.games;

import java.util.ArrayList;

public class ConnectFourState extends BoardState {

    private int[][] state;
    private int player;
    private final int rows = 6;
    private final int cols = 7;

    public ConnectFourState(int[][] state, int player) {
        this.state = state;
        this.player = player;
    }

    public ConnectFourState(ConnectFourState parent) {
        this.state = new int[parent.state.length][parent.state[0].length];
        for (int i = 0; i < parent.state.length; i++) {
            for (int j = 0; j < parent.state[i].length; j++) {
                this.state[i][j] = parent.state[i][j];
            }
        }
        this.player = parent.player;
    }

    @Override
    public BoardState copy() {
        return new ConnectFourState(this);
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
        ArrayList<Move> moves = new ArrayList<>();
        for (int col = 0; col < cols; col++) {
            for (int row = rows - 1; row >= 0; row--) {
                if (state[row][col] == 0) {
                    moves.add(new Move(row, col));
                    break;
                }
            }
        }
        return moves;
    }

    @Override
    public boolean isLegalMove(Move move) {
        int row = move.getX();
        int col = move.getY();
        if (row < 0 || row >= rows || col < 0 || col >= cols) return false;
        if (state[row][col] != 0) return false;
        if (row == rows - 1) return true;
        return state[row + 1][col] != 0;
    }

    @Override
    public void makeMove(Move move) {
        int row = move.getX();
        int col = move.getY();
        state[row][col] = player;
        player = -player;
    }

    @Override
    public int getWinner() {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int cell = state[row][col];
                if (cell == 0) continue;

                if (col + 3 < cols) {
                    boolean win = true;
                    for (int i = 0; i < 4; i++) {
                        if (state[row][col + i] != cell) {
                            win = false;
                            break;
                        }
                    }
                    if (win) return cell;
                }

                if (row + 3 < rows) {
                    boolean win = true;
                    for (int i = 0; i < 4; i++) {
                        if (state[row + i][col] != cell) {
                            win = false;
                            break;
                        }
                    }
                    if (win) return cell;
                }

                if (row + 3 < rows && col + 3 < cols) {
                    boolean win = true;
                    for (int i = 0; i < 4; i++) {
                        if (state[row + i][col + i] != cell) {
                            win = false;
                            break;
                        }
                    }
                    if (win) return cell;
                }

                if (row - 3 >= 0 && col + 3 < cols) {
                    boolean win = true;
                    for (int i = 0; i < 4; i++) {
                        if (state[row - i][col + i] != cell) {
                            win = false;
                            break;
                        }
                    }
                    if (win) return cell;
                }
            }
        }
        return 0;
    }

    @Override
    public boolean isGameOver() {
        if (getWinner() != 0) {
            return true;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (state[i][j] == 0) {
                    return false;
                }
            }
        }
        return true;
    }

    public void display() {
        // Print column headers
        System.out.print("  ");
        for (int j = 0; j < cols; j++) {
            System.out.print(j + " ");
        }
        System.out.println();
        for (int i = 0; i < rows; i++) {
            System.out.print(i + " ");
            for (int j = 0; j < cols; j++) {
                char symbol;
                if (state[i][j] == 1) {
                    symbol = 'X';
                } else if (state[i][j] == -1) {
                    symbol = 'O';
                } else {
                    symbol = '.';
                }
                System.out.print(symbol + " ");
            }
            System.out.println();
        }
    }

    @Override
    public int getScore() {
        int score = 0;
        for (int i = 0; i < state.length; i++) {
            for (int j = 0; j < state[i].length; j++) {
                score += state[i][j];
            }
        }
        return score;
    }

    @Override
    public int boardEval() {
        int ans = 0;
        for (int i = 0; i < 6; i++) for (int j = 0; j < 7; j++) if (
            state[i][j] == 1
        ) {
            ans += getBoardPosition()[i][j];
        } else if (state[i][j] == -1) {
            ans -= getBoardPosition()[i][j];
        }
        return ans;
    }

    public int[][] getBoardPosition() {
        return new int[][] {
            { 3, 4, 5, 7, 5, 4, 3 },
            { 4, 6, 8, 10, 8, 6, 4 },
            { 5, 8, 11, 13, 11, 8, 5 },
            { 5, 8, 11, 13, 11, 8, 5 },
            { 4, 6, 8, 10, 8, 6, 4 },
            { 3, 4, 5, 7, 5, 4, 3 },
        };
    }
}

package tests.java.agents;

import java.io.*;
import tests.java.games.BoardState;
import tests.java.games.ConnectFourState;
import tests.java.games.HexState;
import tests.java.games.Move;
import tests.java.games.OthelloState;

public class MCTSRunner {

    private BufferedReader in;

    private String name;
    private int player = 1;
    private int turn = 0;
    private int size1;
    private int size2;
    private int iterations;

    public MCTSRunner(
        String name,
        int player,
        int size1,
        int size2,
        int iterations
    ) {
        this.name = name;
        this.player = player;
        this.size1 = size1;
        this.size2 = size2;
        this.iterations = iterations;
    }

    private String getMessage() throws IOException {
        return in.readLine();
    }

    private void sendMessage(String msg) {
        System.out.print(msg + "\n");
        System.out.flush();
    }

    public void run() {
        in = new BufferedReader(new InputStreamReader(System.in));

        while (true) {
            // receive messages
            try {
                String msg = getMessage();
                if (msg != null) {
                    boolean res = interpretMessage(msg);
                    if (res == false) break;
                }
            } catch (IOException e) {
                System.out.println("ERROR: Could not establish I/O.");
                return;
            }
        }
    }

    private boolean interpretMessage(String s) {
        String[] msg = s.strip().split(";");
        String board = msg[0];
        if (name.equals("Hex")) {
            turn = Integer.parseInt(msg[1]);
        }
        makeMove(board);
        return false;
    }

    private void makeMove(String board) {
        String[] lines = board.split(",");
        int[][] boardInt = new int[this.size1][this.size2];
        for (int i = 0; i < this.size1; i++) {
            for (int j = 0; j < this.size2; j++) {
                if (lines[i].charAt(j) == '0') {
                    boardInt[i][j] = 0;
                } else if (lines[i].charAt(j) == '1') {
                    boardInt[i][j] = 1;
                } else {
                    boardInt[i][j] = -1;
                }
            }
        }

        BoardState boardState;
        if (name.equals("ConnectFour")) {
            boardState = new ConnectFourState(boardInt, player);
        } else if (name.equals("Othello")) {
            boardState = new OthelloState(boardInt, player);
        } else if (name.equals("Hex")) {
            boardState = new HexState(boardInt, player, turn);
        } else {
            System.out.println("ERROR: Invalid game name.");
            return;
        }

        Node root = new Node(boardState, null, null);
        MCTS mcts = new MCTS();
        Move bestMove = mcts.bestMove(root, iterations);

        sendMessage(bestMove.getX() + "," + bestMove.getY());
    }

    public static void main(String[] args) {
        String gameName = args[0];
        String stringPlayer = args[1];
        String stringSize1 = args[2];
        String stringSize2 = args[3];
        String stringIterations = args[4];
        int player = Integer.parseInt(stringPlayer);
        int size1 = Integer.parseInt(stringSize1);
        int size2 = Integer.parseInt(stringSize2);
        int iterations = Integer.parseInt(stringIterations);
        MCTSRunner agent = new MCTSRunner(
            gameName,
            player,
            size1,
            size2,
            iterations
        );
        agent.run();
    }
}

package tests.java.games;

public class Move {
    
    private int x;
    private int y;

    public Move(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    @Override
    public boolean equals(Object move) {
        if (move instanceof Move) {
            Move moved = (Move) move;
            return (this.x == moved.getX()) && (this.y == moved.getY());
        }
        return false;
    }
}

from src.games.game import Game
import numpy as np


class Othello(Game):
    def __init__(self) -> None:
        super().__init__(8, 8)
        self.reset()
        self.directions = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ])

    def create_game(self) -> "Othello":
        return Othello()
    
    def reset(self) -> None:
        self.state = np.zeros((self.size1, self.size2))
        self.state[3,3] = -1
        self.state[3,4] = 1
        self.state[4,3] = 1
        self.state[4,4] = -1
        self.set_player(1)
        self._legal_moves_cache = None

    def make_move(self, action: int) -> None:
        self._legal_moves_cache = None
        if action == self.size1 * self.size2:
            self.current_player = -self.current_player
            return
        row, col = divmod(action, self.size2)
        self.state[row,col] = self.current_player
        self.flip_pieces(row, col)
        self.current_player = -self.current_player

    def flip_pieces(self, row: int, col: int) -> None:
        for dx, dy in self.directions:
            self._flip_direction(row, col, dx, dy)

    def _flip_direction(self, row: int, col: int, dr: int, dc: int) -> None:
        pieces_to_flip = []
        r, c = row + dr, col + dc
        
        # Collect all potential pieces to flip
        while 0 <= r < self.size1 and 0 <= c < self.size2 and self.state[r, c] == -self.current_player:
            pieces_to_flip.append((r, c))
            r += dr
            c += dc
        
        # If we found a valid line (ending with our piece), flip all collected pieces
        if 0 <= r < self.size1 and 0 <= c < self.size2 and self.state[r, c] == self.current_player and pieces_to_flip:
            for flip_r, flip_c in pieces_to_flip:
                self.state[flip_r, flip_c] = self.current_player


    def get_legal_moves(self) -> np.ndarray:
        if self._legal_moves_cache is None:
            self._legal_moves_cache = self.compute_legal_moves()
        return self._legal_moves_cache
    
    def compute_legal_moves(self) -> np.ndarray:
        legal_moves = []
        
        empty_positions = np.argwhere(self.state == 0)
        
        for row, col in empty_positions:
            action = row * self.size2 + col
            if self.is_legal_move(action):
                legal_moves.append(action)
        
        if not legal_moves:
            legal_moves.append(self.size1 * self.size2)
            
        return np.array(legal_moves)

    def is_legal_move(self, action: int) -> bool:
        if action == self.size1 * self.size2:

            empty_positions = np.argwhere(self.state == 0)
            for row, col in empty_positions:
                if self.is_legal_move_position(row, col):
                    return False
            return True
        row, col = divmod(action, self.size2)
        return self.is_legal_move_position(row, col)
    
    def is_legal_move_position(self, row: int, col: int) -> bool:
        if not (0 <= row < self.size1 and 0 <= col < self.size2) or self.state[row, col] != 0:
            return False
            
        for dr, dc in self.directions:
            if self.is_legal_direction(row, col, dr, dc):
                return True
        return False

    def is_legal_direction(self, row: int, col: int, dr: int, dc: int) -> bool:
        r, c = row + dr, col + dc
        
        if not (0 <= r < self.size1 and 0 <= c < self.size2) or self.state[r, c] != -self.current_player:
            return False
            
        r += dr
        c += dc
        while 0 <= r < self.size1 and 0 <= c < self.size2 and self.state[r, c] == -self.current_player:
            r += dr
            c += dc
            
        return 0 <= r < self.size1 and 0 <= c < self.size2 and self.state[r, c] == self.current_player

    def get_winner(self) -> int:
        if not self.is_game_over():
            return 0
        score = np.sum(self.state)
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def is_game_over(self) -> bool:
        if len(self.get_legal_moves()) == 1 and self.get_legal_moves()[0] == self.size1 * self.size2:
            self.current_player = -self.current_player
            self._legal_moves_cache = None
    
            opponent_must_pass = len(self.get_legal_moves()) == 1 and self.get_legal_moves()[0] == self.size1 * self.size2
            self.current_player = -self.current_player
            self._legal_moves_cache = None
            return opponent_must_pass
        return False

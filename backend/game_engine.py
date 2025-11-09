import chess
import chess.pgn
from typing import Optional, List

class GameEngine:
    def __init__(self):
        self.board = chess.Board()
        self.moves_played: List[chess.Move] = []

    def reset(self):
        self.board.reset()
        self.moves_played = []

    def legal_moves(self):
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> bool:
        """Push move if legal and record it. Return True if made."""
        if move in self.board.legal_moves:
            self.board.push(move)
            self.moves_played.append(move)
            return True
        return False
    
    def is_game_over(self) -> bool:
        return self.board.is_game_over()
    
    def result(self) -> Optional[str]:
        if not self.board.is_game_over():
            return None
        return self.board.result()
    
    def current_player(self) -> bool:
        return self.board.turn == chess.WHITE
    
    def board_fen(self) -> str:
        return self.board.fen()
    
    def export_pgn(self, filename: str):
        game = chess.pgn.Game()
        node = game
        for mv in self.moves_played:
            node = node.add_variation(mv)
        # set result header if available
        res = None
        if self.board.is_game_over():
            res = self.board.result()
            game.headers["Result"] = res
        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(game))

    def get_move_list(self) -> List[str]:
        """Return list of SAN strings for the moves played so far."""
        temp = chess.Board()
        san_list: List[str] = []
        for mv in self.moves_played:
            try:
                san = temp.san(mv)
            except Exception:
                # fallback to UCI if SAN generation fails
                san = mv.uci()
            san_list.append(san)
            temp.push(mv)
        return san_list




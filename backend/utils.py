import torch
import chess
from backend.config import ACTION_SIZE, DEVICE

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((12,8,8), dtype = torch.float32)
    piece_map = board.piece_map()
    piece_to_plane = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    #piece_map: dict[int, chess.Piece] = board.piece_map()
    for square,piece in piece_map.items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece_to_plane[piece.piece_type]
        row = 7 - (square//8)
        col = square%8
        planes[plane, row, col] = 1.0

    return planes.unsqueeze(0).to(DEVICE)


def move_to_action(move: chess.Move) -> int:
    if move.promotion is None:
        return move.from_square * 64 + move.to_square
    else:
        promo_map = {chess.QUEEN:0, chess.ROOK:1, chess.BISHOP:2, chess.KNIGHT:3}
        return 64*64 + move.from_square*4 + promo_map[move.promotion]

def action_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    if action_idx < 64*64:
        from_sq = action_idx // 64
        to_sq = action_idx % 64
        move = chess.Move(from_sq, to_sq)
    else:
        x = action_idx - 64*64
        from_sq = x // 4
        promo_map = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        promotion = promo_map[x % 4]

        move = None
        for m in board.legal_moves:
            if m.from_square == from_sq and m.promotion == promotion:
                move = m
        
        if move is None:
            # fallback to any legal move (should be rare)
            move = list(board.legal_moves)[0]
    return move

def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(ACTION_SIZE, dtype = torch.bool, device= DEVICE)
    for move in board.legal_moves:
        idx = move_to_action(move)
        if idx < ACTION_SIZE:
            mask[idx] = True
    return mask

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    return torch.softmax(masked_logits, dim=0)
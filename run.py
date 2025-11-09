# run.py
import eel
import os
import random
import torch
import chess
import atexit
import json
import time

from backend.game_engine import GameEngine
from backend.model import load_model
from backend.trainer import Trainer
from backend.config import DEVICE, EPSILON, MODEL_DIR, SAVE_EVERY_N_GAMES
from backend.utils import board_to_tensor, legal_move_mask, masked_softmax, action_to_move, move_to_action

# ensure folders exist before loading/saving models
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

eel.init("web")  # your web folder

# global objects
MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.pth")
MODEL = load_model(MODEL_PATH)  # load if present, load_model handles missing file
TRAINER = Trainer(MODEL)

# save model on normal exit
atexit.register(lambda: TRAINER.save_model("latest_model.pth"))

# instantiate game engine (was missing)
ENGINE = GameEngine()

# ---- simple training bookkeeping ----
GAME_STATES_W = []
GAME_ACTIONS_W = []
GAME_STATES_B = []
GAME_ACTIONS_B = []
GAMES_SINCE_TRAIN = 0
TRAIN_EVERY_N_GAMES = 1   # tune as needed
SAVE_EVERY_N_GAMES = 5    # tune as needed

# performance logging file
PERF_FILE = os.path.join("data", "perf.json")
# load existing perf or initialize
if os.path.exists(PERF_FILE):
    try:
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            PERF = json.load(f)
    except Exception:
        PERF = {"games": [], "training": []}
else:
    PERF = {"games": [], "training": []}

def _save_perf():
    os.makedirs("data", exist_ok=True)
    try:
        with open(PERF_FILE, "w", encoding="utf-8") as f:
            json.dump(PERF, f, indent=2)
    except Exception as e:
        print("Failed saving perf:", e)

def record_game_result(result_str: str):
    # result_str like "1-0", "0-1", "1/2-1/2"
    entry = {
        "ts": int(time.time()),
        "result": result_str,
        "moves": len(ENGINE.moves_played) if hasattr(ENGINE, "moves_played") else len(ENGINE.get_move_list()),
    }
    PERF["games"].append(entry)
    # keep last 500 entries to avoid growing forever
    PERF["games"] = PERF["games"][-500:]
    _save_perf()

def record_training(loss):
    entry = {
        "ts": int(time.time()),
        "loss": None if loss is None else float(loss)
    }
    PERF["training"].append(entry)
    PERF["training"] = PERF["training"][-500:]
    _save_perf()

# Start background training so the model improves continuously.
# We start the trainer after record_training exists so the callback is available.
TRAINER.start_background_training(interval=1.0, save_every=SAVE_EVERY_N_GAMES, loss_callback=record_training)

# ensure background trainer stops and model saved on exit
atexit.register(lambda: (TRAINER.stop_background_training(), TRAINER.save_model("latest_model.pth")))

@eel.expose
def get_performance():
    # return a small summary + last few records
    games = PERF.get("games", [])
    training = PERF.get("training", [])
    wins = sum(1 for g in games if g["result"] == "1-0")
    losses = sum(1 for g in games if g["result"] == "0-1")
    draws = sum(1 for g in games if g["result"] in ("1/2-1/2", "1/2-1/2"))
    recent_games = games[-10:]
    recent_training = training[-5:]
    return {
        "counts": {"wins": wins, "losses": losses, "draws": draws, "total": len(games)},
        "recent_games": recent_games,
        "recent_training": recent_training
    }

# helper: choose an action for a board (same logic as earlier choose_action_for_board)
def choose_action_for_board(board: chess.Board, model, epsilon: float = EPSILON, temperature: float = 1.0):
    with torch.no_grad():
        state_t = board_to_tensor(board)          # (1,12,8,8) on DEVICE
        logits = model(state_t).squeeze(0)       # (ACTION_SIZE,)
        mask = legal_move_mask(board)            # boolean mask on DEVICE

        if mask.sum().item() == 0:
            return None, None

        probs = masked_softmax(logits, mask)

        # epsilon-greedy
        if random.random() < epsilon:
            legal_idxs = torch.nonzero(mask).squeeze(-1).tolist()
            if isinstance(legal_idxs, int):
                idx = int(legal_idxs)
            else:
                idx = random.choice(legal_idxs)
        else:
            idx = torch.multinomial(probs, num_samples=1).item()

        move = action_to_move(int(idx), board)
        # fallback if action_to_move produced illegal
        if move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
            idx = move_to_action(move)

    return int(idx), move

# -----------------------
# Eel-exposed API
# -----------------------
@eel.expose
def get_board_fen():
    """Return current board FEN for frontend rendering."""
    return ENGINE.board_fen()

@eel.expose
def get_moves():
    """Return list of moves in SAN notation (strings)."""
    return ENGINE.get_move_list()

@eel.expose
def reset_game():
    global GAME_STATES_W, GAME_ACTIONS_W, GAME_STATES_B, GAME_ACTIONS_B
    ENGINE.reset()
    GAME_STATES_W, GAME_ACTIONS_W, GAME_STATES_B, GAME_ACTIONS_B = [], [], [], []
    return ENGINE.board_fen()

@eel.expose
def make_human_move(move_str: str):
    """
    Expect UCI move like 'e2e4' or UCI with promotion 'e7e8q'.
    """
    global GAME_STATES_W, GAME_ACTIONS_W, GAME_STATES_B, GAME_ACTIONS_B, GAMES_SINCE_TRAIN

    try:
        # try UCI
        move = chess.Move.from_uci(move_str)
    except Exception:
        try:
            move = ENGINE.board.parse_san(move_str)
        except Exception:
            return "invalid"

    if move not in ENGINE.board.legal_moves:
        return "invalid"

    # record state/action for training (state BEFORE move)
    state_t = board_to_tensor(ENGINE.board).detach().cpu()
    aidx = move_to_action(move)
    if ENGINE.board.turn == chess.WHITE:
        GAME_STATES_W.append(state_t)
        GAME_ACTIONS_W.append(aidx)
    else:
        GAME_STATES_B.append(state_t)
        GAME_ACTIONS_B.append(aidx)

    ENGINE.make_move(move)

    if ENGINE.is_game_over():
        # compute rewards: +1 winner, -1 loser, 0.5 draw
        res = ENGINE.result()
        if res == "1-0":
            reward_w, reward_b = 1.0, -1.0
        elif res == "0-1":
            reward_w, reward_b = -1.0, 1.0
        else:
            reward_w, reward_b = 0.5, 0.5

        # store episodes into replay buffer (if any moves)
        if GAME_STATES_W and GAME_ACTIONS_W:
            TRAINER.store_game(GAME_STATES_W, GAME_ACTIONS_W, reward_w)
        if GAME_STATES_B and GAME_ACTIONS_B:
            TRAINER.store_game(GAME_STATES_B, GAME_ACTIONS_B, reward_b)

        GAME_STATES_W, GAME_ACTIONS_W, GAME_STATES_B, GAME_ACTIONS_B = [], [], [], []
        GAMES_SINCE_TRAIN += 1

        # train / save periodically
        if GAMES_SINCE_TRAIN >= TRAIN_EVERY_N_GAMES:
            loss = TRAINER.train_step()
            # record training loss
            record_training(loss)
            GAMES_SINCE_TRAIN = 0
        if (GAMES_SINCE_TRAIN % SAVE_EVERY_N_GAMES) == 0:
            TRAINER.save_model("latest_model.pth")

        # record game perf
        record_game_result(res)

        return f"game_over:{res}"

    return "ok"

@eel.expose
def ai_move():
    """
    Ask the model to pick and play a move, returns move.uci() or game_over.
    """
    global GAME_STATES_W, GAME_ACTIONS_W, GAME_STATES_B, GAME_ACTIONS_B, GAMES_SINCE_TRAIN

    if ENGINE.is_game_over():
        return f"game_over:{ENGINE.result()}"

    idx, move = choose_action_for_board(ENGINE.board, MODEL)
    if move is None:
        return "invalid"

    # record AI's state/action (board before AI move)
    state_t = board_to_tensor(ENGINE.board).detach().cpu()
    if ENGINE.board.turn == chess.WHITE:
        GAME_STATES_W.append(state_t)
        GAME_ACTIONS_W.append(int(idx))
    else:
        GAME_STATES_B.append(state_t)
        GAME_ACTIONS_B.append(int(idx))

    ENGINE.make_move(move)

    # persist model periodically (also trainer may already save)
    try:
        TRAINER.save_model("latest_model.pth")
    except Exception:
        pass

    if ENGINE.is_game_over():
        res = ENGINE.result()
        if res == "1-0":
            reward_w, reward_b = 1.0, -1.0
        elif res == "0-1":
            reward_w, reward_b = -1.0, 1.0
        else:
            reward_w, reward_b = 0.5, 0.5

        if GAME_STATES_W and GAME_ACTIONS_W:
            TRAINER.store_game(GAME_STATES_W, GAME_ACTIONS_W, reward_w)
        if GAME_STATES_B and GAME_ACTIONS_B:
            TRAINER.store_game(GAME_STATES_B, GAME_ACTIONS_B, reward_b)

        GAME_STATES_W, GAME_ACTIONS_W, GAME_STATES_B, GAME_ACTIONS_B = [], [], [], []
        GAMES_SINCE_TRAIN += 1
        if GAMES_SINCE_TRAIN >= TRAIN_EVERY_N_GAMES:
            loss = TRAINER.train_step()
            record_training(loss)
            GAMES_SINCE_TRAIN = 0

        # record game perf
        record_game_result(res)

        return f"game_over:{res}"

    return move.uci()

# -----------------------
# Start Eel UI
# -----------------------
if __name__ == "__main__":
    # ensure models & data folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    # Option A — use system default browser (recommended)
    eel.start("index.html", size=(900, 900), block=True, mode="system")

    # Option B — or point to a specific browser executable (Windows example)
    # chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    # eel.start("index.html", size=(900, 900), block=True, mode=chrome_path)

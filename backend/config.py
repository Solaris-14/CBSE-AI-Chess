import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 2000
UPDATE_EVERY_N_GAMES = 1
EPSILON = 0.05

# ACTION space:
#  - 64*64 for normal from->to (4096)
#  - 64*4  for promotions (256)
ACTION_SIZE = 64*64 + 64*4  # 4096 + 256 = 4352

# Use project-root relative directories (the app creates these at runtime)
MODEL_DIR = "models"
DATA_DIR = "data"

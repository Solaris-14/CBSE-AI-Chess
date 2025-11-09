import threading
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim

from typing import List
from backend.config import (
    LEARNING_RATE,
    BATCH_SIZE,
    MODEL_DIR,
    DEVICE
)

from backend.utils import masked_softmax
from backend.replay_buffer import ReplayBuffer
from backend.model import ChessPolicyNet
import os

class Trainer:
    def __init__(self, model: ChessPolicyNet):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.buffer = ReplayBuffer()
        # background training controls
        self._stop_event = None
        self._bg_thread = None

    def store_game(self, states: List[torch.Tensor], actions: List[int], reward: float):
        # states: list of tensors (1,12,8,8) or (12,8,8) — normalize to (1,12,8,8)
        norm_states = []
        for s in states:
            if s.dim() == 3:
                norm_states.append(s.unsqueeze(0))
            else:
                norm_states.append(s)
        self.buffer.add(norm_states, actions, float(reward))

    def train_step(self):
        if len(self.buffer) == 0:
            return None
        
        states, actions, rewards = self.buffer.sample(BATCH_SIZE)
        if len(states) == 0:
            return None
        
        # states is list of tensors shaped (1,12,8,8) -> concat -> (batch,12,8,8)
        states_tensor = torch.cat(states, dim=0).to(DEVICE)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

        self.model.train()
        logits = self.model(states_tensor)

        base_losses = self.loss_fn(logits, actions_tensor)          # (batch,)
        weighted_losses = base_losses * rewards_tensor              # reward-weighted

        loss = weighted_losses.mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
    
    def save_model(self,filename = "latest_model.pth"):
        path = os.path.join(MODEL_DIR, filename)
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, filename = "latest_model.pth"):
        path = os.path.join(MODEL_DIR, filename)
        try:
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"Loaded model from {path}")
        except FileNotFoundError:
            print("⚠ No saved model found. Starting fresh.")

    # Background training control ------------------------------------------------
    def _train_loop(self, interval: float, save_every: int, loss_callback):
        i = 0
        while not self._stop_event.is_set():
            loss = self.train_step()
            if loss is not None and loss_callback is not None:
                try:
                    loss_callback(loss)
                except Exception:
                    pass
            i += 1
            if save_every and (i % save_every) == 0:
                try:
                    self.save_model()
                except Exception:
                    pass
            time.sleep(interval)

    def start_background_training(self, interval: float = 1.0, save_every: int = 10, loss_callback=None):
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._bg_thread = threading.Thread(target=self._train_loop, args=(interval, save_every, loss_callback), daemon=True)
        self._bg_thread.start()
        print("Background trainer started.")

    def stop_background_training(self):
        if self._stop_event is None:
            return
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=2.0)
        print("Background trainer stopped.")
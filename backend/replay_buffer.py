from collections import deque, namedtuple
import random
from backend.config import REPLAY_BUFFER_SIZE

Episode = namedtuple("Episode", ["states", "actions", "reward"])

class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, states: list, actions: list, reward: float):
        episode = Episode(states=states, actions=actions, reward=reward)
        self.buffer.append(episode)

    def sample(self, batch_size: int):
        if len(self.buffer) == 0 or batch_size <= 0:
            return [], [], []

        # if we have enough unique episodes, sample without replacement,
        # otherwise allow repeats with choices
        if len(self.buffer) >= batch_size:
            sampled_episodes = random.sample(self.buffer, k=batch_size)
        else:
            sampled_episodes = random.choices(self.buffer, k=batch_size)

        sampled_states, sampled_actions, sampled_rewards = [], [], []

        for ep in sampled_episodes:
            if not ep.states:
                continue
            idx = random.randrange(len(ep.states))
            sampled_states.append(ep.states[idx])
            sampled_actions.append(ep.actions[idx])
            sampled_rewards.append(ep.reward)

        return sampled_states, sampled_actions, sampled_rewards

    def __len__(self):
        return len(self.buffer)


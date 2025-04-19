# 实现DQN算法

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from env import DaVinciCodeGameEnvironment
import os

def create_masked_state(state_vector, max_size):
    """Creates a masked state vector with a binary mask."""
    mask = [1] * len(state_vector) + [0] * (max_size - len(state_vector))
    if len(state_vector) < max_size:
        padding = [0] * (max_size - len(state_vector))
        state_vector = np.concatenate([state_vector, padding])
    return state_vector, mask

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, action_size)

    def forward(self, x, mask=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if mask is not None:
            x = x * mask  # Apply the mask
        x = self.fc3(x)
        return x

# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 128

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, filepath):
        """Saves the model and optimizer state to a checkpoint file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Loads the model and optimizer state from a checkpoint file."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Checkpoint loaded from {filepath}")
        else:
            print(f"No checkpoint found at {filepath}")

def encode_hand(hand, mapping):
    """Encodes a hand of strings into numeric values using a mapping."""
    encoded_hand = []
    for card in hand:
        if card in mapping:
            encoded_hand.append(mapping[card])
        elif card == "B?":
            encoded_hand.append(-2)  # Assign -2 for hidden black cards
        elif card == "W?":
            encoded_hand.append(-3)  # Assign -3 for hidden white cards
        else:
            encoded_hand.append(-1)  # Default for unknown cards
    return encoded_hand

# 训练DQN智能体
if __name__ == "__main__":
    env = DaVinciCodeGameEnvironment()
    # Create a mapping for string cards to numeric values
    unique_cards = set(env.get_all_possible_cards())  # Assuming the environment provides this method
    print(unique_cards)
    card_mapping = {card: idx for idx, card in enumerate(unique_cards)}

    
    max_state_size = 28 # 12*2+2+2

    state = env.reset()
    state_size = max_state_size  # Use the fixed maximum state size
    action_size = len(env._get_legal_actions())  # 动作空间大小
    agent = DQNAgent(state_size, action_size)
    episodes = 10000

    checkpoint_dir = "d:\\ai4s\\Davinci-Code-Agent\\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "dqn_checkpoint.pth")

    # Load checkpoint if it exists
    agent.load_checkpoint(checkpoint_path)

    for e in range(episodes):
        state = env.reset()
        state_vector = np.concatenate([
            np.array(encode_hand(state['hand'], card_mapping), dtype=np.float32),
            np.array(encode_hand(state['opponent_hand_visible'], card_mapping), dtype=np.float32),
            np.array([state['deck_size'], state['current_player']], dtype=np.float32)
        ])
        state_vector, mask = create_masked_state(state_vector, max_state_size)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state_vector)
            legal_actions = env._get_legal_actions()
            if action >= len(legal_actions):
                action = random.choice(range(len(legal_actions)))  # 防止非法动作
            chosen_action = legal_actions[action]

            next_state, reward, done, _ = env.step(chosen_action)
            next_state_vector = np.concatenate([
                np.array(encode_hand(next_state['hand'], card_mapping), dtype=np.float32),
                np.array(encode_hand(next_state['opponent_hand_visible'], card_mapping), dtype=np.float32),
                np.array([next_state['deck_size'], next_state['current_player']], dtype=np.float32)
            ])
            next_state_vector, mask = create_masked_state(next_state_vector, max_state_size)
            agent.remember(state_vector, action, reward, next_state_vector, done)
            state_vector = next_state_vector
            total_reward += reward

            if done:
                print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")
                agent.update_target_model()

        agent.replay()

        # Save checkpoint every 100 episodes
        if (e + 1) % 1000 == 0:
            agent.save_checkpoint(checkpoint_path)

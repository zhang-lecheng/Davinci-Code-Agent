# 实现 PPO 算法
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env import DaVinciCodeGameEnvironment
from DQN import create_masked_state, encode_hand
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.shared_fc = nn.Linear(state_size, 256)
        self.actor_fc = nn.Linear(256, action_size)
        self.critic_fc = nn.Linear(256, 1)

    def forward(self, x, mask=None):
        x = torch.relu(self.shared_fc(x))
        action_logits = self.actor_fc(x)
        if mask is not None:
            action_logits = action_logits + (mask * -1e9)  # Apply mask
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic_fc(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.0003, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(state_size, action_size)
        self.policy_old = ActorCritic(state_size, action_size)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory, action_mask):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
        action_probs, _ = self.policy_old(state, mask=action_mask_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def update(self, memory):
        # Convert memory to tensors
        old_states = torch.cat(memory.states).detach()
        old_actions = torch.cat(memory.actions).detach()
        old_logprobs = torch.cat(memory.logprobs).detach()

        # Compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Optimize policy for k epochs
        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            state_values = state_values.squeeze()

            # Compute ratios
            ratios = torch.exp(logprobs - old_logprobs)

            # Compute surrogate loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * entropy.mean()

            # Update policy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_checkpoint(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {filepath}")
        else:
            print(f"No checkpoint found at {filepath}")

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

if __name__ == "__main__":
    env = DaVinciCodeGameEnvironment()
    unique_cards = set(env.get_all_possible_cards())
    card_mapping = {card: idx for idx, card in enumerate(unique_cards)}

    max_state_size = 28  # 12*2+2+2
    state_size = max_state_size
    action_size = len(env.get_all_possible_cards()) + 1  # +1 for 'place' action
    agent = PPOAgent(state_size, action_size)
    memory = Memory()

    checkpoint_dir = "d:\\ai4s\\Davinci-Code-Agent\\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "ppo_checkpoint.pth")

    # Load checkpoint if it exists
    agent.load_checkpoint(checkpoint_path)

    episodes = 10000
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
            legal_actions = env._get_legal_actions()
            action_mask = np.zeros(agent.action_size, dtype=np.float32)
            action_mask[:len(legal_actions)] = 0  # Valid actions
            action_mask[len(legal_actions):] = 1  # Invalid actions

            action = agent.select_action(state_vector, memory, action_mask)
            chosen_action = legal_actions[action]

            next_state, reward, done, _ = env.step(chosen_action)
            next_state_vector = np.concatenate([
                np.array(encode_hand(next_state['hand'], card_mapping), dtype=np.float32),
                np.array(encode_hand(next_state['opponent_hand_visible'], card_mapping), dtype=np.float32),
                np.array([next_state['deck_size'], next_state['current_player']], dtype=np.float32)
            ])
            next_state_vector, mask = create_masked_state(next_state_vector, max_state_size)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state_vector = next_state_vector
            total_reward += reward

        agent.update(memory)
        memory.clear_memory()

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

        # Save checkpoint every 1000 episodes
        if (e + 1) % 1000 == 0:
            agent.save_checkpoint(checkpoint_path)

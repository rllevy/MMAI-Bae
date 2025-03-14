import numpy as np
import time
import pickle
import os
from warehouse_liam import WarehouseEnv

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay
        self.q_table = {}
        
    def get_state_key(self, observation):
        # Convert the observation to a hashable representation
        # We'll use agent position and whether it's carrying an item
        agent_pos = tuple(observation['agent_position'])
        carrying = observation['carrying_item']
        
        # Also include positions of items that haven't been picked up or delivered
        item_positions = []
        for item_pos in observation['items']:
            if item_pos[0] >= 0:  # Not delivered
                item_positions.append(tuple(item_pos))
        
        # Sort item positions to ensure consistent state representation
        item_positions = tuple(sorted(item_positions))
        
        return (agent_pos, carrying, item_positions)
    
    def get_action(self, observation):
        state_key = self.get_state_key(observation)
        
        # If this state hasn't been seen before, initialize its Q-values
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state_key])  # Exploit
    
    def update(self, observation, action, reward, next_observation, done):
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)
        
        # If next state hasn't been seen before, initialize its Q-values
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        target_q = reward + (0 if done else self.discount_factor * max_next_q)
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        if done:
            self.exploration_rate *= self.exploration_decay
    
    def save(self, filepath):
        """Save the agent to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay
            }, f)
        print(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, action_space):
        """Load an agent from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(
            action_space=action_space,
            learning_rate=data['learning_rate'],
            discount_factor=data['discount_factor'],
            exploration_rate=data['exploration_rate'],
            exploration_decay=data['exploration_decay']
        )
        agent.q_table = data['q_table']
        print(f"Agent loaded from {filepath}")
        return agent

def train_agent(episodes=1000, render_every=100, save_path=None):
    env = WarehouseEnv(width=10, height=10, render_mode=None, num_items=1)
    agent = QLearningAgent(env.action_space)
    
    # Define action meanings
    action_meanings = {
        0: "Move Left",
        1: "Move Right",
        2: "Move Up",
        3: "Move Down",
        4: "Pickup Item",
        5: "Drop Item"
    }
    
    # Track metrics
    rewards_per_episode = []
    steps_per_episode = []
    items_delivered_per_episode = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Render occasionally to see progress
        render_this_episode = (episode % render_every == 0)
        if render_this_episode:
            env_render = WarehouseEnv(width=10, height=10, render_mode="human", num_items=1)
            render_obs, _ = env_render.reset()
        
        while not done:
            # Get action from agent
            action = agent.get_action(observation)
            
            # Take action in environment
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Update agent's knowledge
            agent.update(observation, action, reward, next_observation, done or truncated)
            
            # Render if needed
            if render_this_episode:
                render_action = agent.get_action(render_obs)
                render_obs, _, render_done, render_truncated, _ = env_render.step(render_action)
                print(f"Episode {episode}, Step {steps}, Action: {action_meanings[render_action]}, Reward: {reward}")
                env_render.render()
                time.sleep(0.2)
                if render_done or render_truncated:
                    break
            
            # Update tracking variables
            observation = next_observation
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Close render environment if used
        if render_this_episode:
            env_render.close()
        
        # Record metrics
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        items_delivered_per_episode.append(info['items_delivered'])
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            avg_steps = np.mean(steps_per_episode[-10:])
            avg_items = np.mean(items_delivered_per_episode[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Avg Items: {avg_items:.2f}, Epsilon: {agent.exploration_rate:.4f}")
        
        # Save agent periodically
        if save_path and episode > 0 and episode % 100 == 0:
            checkpoint_path = f"{save_path}_episode_{episode}.pkl"
            agent.save(checkpoint_path)
    
    # Save final agent
    if save_path:
        agent.save(f"{save_path}_final.pkl")
    
    env.close()
    return agent, rewards_per_episode, steps_per_episode, items_delivered_per_episode

def test_agent(agent, episodes=5):
    env = WarehouseEnv(width=10, height=10, render_mode="human", num_items=1)
    
    # Define action meanings
    action_meanings = {
        0: "Move Left",
        1: "Move Right",
        2: "Move Up",
        3: "Move Down",
        4: "Pickup Item",
        5: "Drop Item"
    }
    
    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n=== TESTING EPISODE {episode+1} ===")
        env.render()
        time.sleep(1)
        
        while not done:
            # Get action from agent (no exploration)
            state_key = agent.get_state_key(observation)
            if state_key in agent.q_table:
                action = np.argmax(agent.q_table[state_key])
            else:
                action = env.action_space.sample()
            
            # Take action
            observation, reward, done, truncated, info = env.step(action)
            
            # Show result
            print(f"Step {steps+1}, Action: {action_meanings[action]}, Reward: {reward}")
            env.render()
            time.sleep(0.5)
            
            # Update tracking
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        print(f"Episode {episode+1} completed with total reward: {total_reward:.2f}")
        print(f"Items delivered: {info['items_delivered']}/{env.num_items}")
    
    env.close()

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Train or load agent
    load_agent = False  # Set to True to load a pre-trained agent
    agent_path = os.path.join(models_dir, "warehouse_agent")
    
    if load_agent:
        # Load a pre-trained agent
        env = WarehouseEnv(width=10, height=10)
        trained_agent = QLearningAgent.load(f"{agent_path}_final.pkl", env.action_space)
        env.close()
    else:
        # Train a new agent
        print("Training Q-learning agent...")
        trained_agent, rewards, steps, items = train_agent(episodes=500, render_every=100, save_path=agent_path)
        print("\nTraining completed!")
        print(f"Final exploration rate: {trained_agent.exploration_rate:.4f}")
        print(f"Q-table size: {len(trained_agent.q_table)} states")
    
    # Test the agent
    print("\nTesting trained agent...")
    test_agent(trained_agent) 
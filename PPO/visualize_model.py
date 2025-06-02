import gymnasium as gym
import torch
from agent import Agent

def visualize_trained_model():
    # Create environment with rendering
    env = gym.make('CartPole-v1', render_mode='human')

    # Create and load the trained agent
    agent = Agent()

    # Run 5 episodes to visualize the trained model
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action from the trained policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action_probs = agent.policy_network.pi(state_tensor)
                action = torch.distributions.Categorical(action_probs).sample().item()

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += reward
            state = next_state

        print(f"Episode {episode + 1} finished with reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    visualize_trained_model()
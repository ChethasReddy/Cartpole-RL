import gym

def evaluate_agent(agent, episodes=5, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    agent.epsilon = 0.0  # disable exploration
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Evaluation Episode {ep}: Total Reward = {total_reward}")
        rewards.append(total_reward)

    env.close()
    return rewards

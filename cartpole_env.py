import gym

def create_env():
    env = gym.make("CartPole-v1", render_mode="human")  # specify render_mode
    observation, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()

if __name__ == "__main__":
    create_env()

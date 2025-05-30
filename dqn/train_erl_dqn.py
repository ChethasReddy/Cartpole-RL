import gym
import matplotlib.pyplot as plt
from erl_dqn import EvolutionManager, DQNAgent
from utils import evaluate_agent
from matplotlib import pyplot as plt

def plot_sample_efficiency(rewards, steps):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards)
    plt.xlabel("Environment Steps")
    plt.ylabel("Best Average Reward")
    plt.title("Sample Efficiency of ERL (DQN)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_erl_dqn(env_name="CartPole-v1", generations=20, population_size=10):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    manager = EvolutionManager(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        agent_class=DQNAgent,
        population_size=population_size,
        elite_frac=0.2,
        mutation_rate=0.05
    )

    best_rewards = []
    env_steps = []

    total_env_steps = 0

    for gen in range(generations):
        steps = manager.evolve()
        total_env_steps += steps

        best_agent = manager.get_best_agent()
        best_reward = best_agent.fitness

        env_steps.append(total_env_steps)
        best_rewards.append(best_reward)

        print(f"Generation {gen + 1}: Best Avg Reward = {best_reward:.2f}, Env Steps = {total_env_steps}")

    plot_sample_efficiency(best_rewards, env_steps)
    env.close()
    return manager.get_best_agent()

if __name__ == "__main__":
    best_agent = train_erl_dqn()
    evaluate_agent(best_agent, episodes=5, render=False)

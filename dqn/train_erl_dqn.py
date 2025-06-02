import gym
import matplotlib.pyplot as plt
import numpy as np
from erl_dqn import EvolutionManager, DQNAgent
from utils import evaluate_agent

def plot_sample_efficiency(rewards, steps):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards)
    plt.xlabel("Environment Steps")
    plt.ylabel("Top-k Average Reward")
    plt.title("Sample Efficiency of ERL (DQN)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_erl_dqn(env_name="CartPole-v1", generations=30, population_size=15, top_k=2):
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

    avg_topk_rewards = []
    env_steps = []

    total_env_steps = 0
    solved_at_step = None
    reach_times = 0

    gen = 0
    while gen < generations and total_env_steps < 100000:
        gen += 1
        steps = manager.evolve()
        total_env_steps += steps

        # Average top-k fitness for smoother reward tracking
        top_k_agents = sorted(manager.population, key=lambda a: a.fitness, reverse=True)[:top_k]
        top_k_avg_reward = np.mean([agent.fitness for agent in top_k_agents])

        if solved_at_step is None and top_k_avg_reward >= 195:
            solved_at_step = total_env_steps
            print(f"✅ Environment solved at step {solved_at_step}")

        if top_k_avg_reward >= 195:
            reach_times += 1

        env_steps.append(total_env_steps)
        avg_topk_rewards.append(top_k_avg_reward)

        print(f"Generation {gen}: Top-{top_k} Avg Reward = {top_k_avg_reward:.2f}, Env Steps = {total_env_steps}")

    print(f"Total generations: {gen}, Total environment steps: {total_env_steps}, Reached 195 reward {reach_times} times")
    plot_sample_efficiency(avg_topk_rewards, env_steps)
    env.close()
    
    return manager.get_best_agent()

if __name__ == "__main__":
    best_agent = train_erl_dqn()
    evaluate_agent(best_agent, episodes=5, render=False)

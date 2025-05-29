import gym
import matplotlib.pyplot as plt
from erl_dqn import EvolutionManager, DQNAgent
from utils import evaluate_agent

def train_erl_dqn(env_name="CartPole-v1", generations=20, population_size=10):
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize Evolution Manager
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

    # Main evolution loop
    for gen in range(generations):
        manager.evolve()
        best_agent = manager.get_best_agent()
        best_reward = best_agent.fitness
        best_rewards.append(best_reward)

        print(f"Generation {gen + 1}: Best Avg Reward = {best_reward:.2f}")

    # Save and plot reward progression
    plt.plot(best_rewards)
    plt.title("ERL (DQN) - Best Average Reward per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.show()

    env.close()
    return manager.get_best_agent()

if __name__ == "__main__":
    best_agent = train_erl_dqn()
    evaluate_agent(best_agent, episodes=5, render=True)

class AgentConfig:
    # Learning
    gamma = 0.99
    plot_every = 10
    update_freq = 1
    k_epoch = 3
    learning_rate = 0.02
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01

    # Memory
    memory_size = 400

    train_cartpole = True

class EvolutionConfig:
    # Evolutionary Algorithm Parameters
<<<<<<< HEAD
    population_size = 10
    num_generations = 20
=======
    population_size = 50
    num_generations = 30
>>>>>>> 3bd563f (made hybrid evolutionary reinforcement Algorithm using PPO)
    mutation_rate = 0.1
    mutation_scale = 0.1
    tournament_size = 5
    elite_size = 1

    # Hybrid Training Parameters
<<<<<<< HEAD
    evolution_epochs = 5  # Number of evolutionary generations before PPO fine-tuning
=======
    evolution_epochs = 3  # Number of evolutionary generations before PPO fine-tuning
>>>>>>> 3bd563f (made hybrid evolutionary reinforcement Algorithm using PPO)
    ppo_epochs = 3  # Number of PPO updates after evolution
    num_ppo_policies = 3  # Number of top policies to fine-tune with PPO

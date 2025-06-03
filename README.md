Cartpole-rl

### 1. Clone repository

```bash
git clone https://github.com/ChethasReddy/Cartpole-RL.git
cd cartpole-rl
```

### 2. Setup the virtual environment

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the environment test

```bash
python cartpole_env.py
```

### 5. Running Evolutionary PPO

The project implements a hybrid Evolutionary-PPO algorithm for training the CartPole agent. To run the training:

```bash
# Navigate to the PPO directory
cd PPO

# Run the training
python run_training.py
```

During training, you'll see:

- Episode numbers
- Generation progress
- Best fitness scores
- PPO episode rewards

The training progress is automatically plotted and saved as 'score.png' in the PPO directory.

To visualize the trained model:

```bash
python visualize_model.py
```

This will show the trained agent balancing the pole in real-time for 5 episodes.

### 6. Running DQN and Evolutionary DQN

You can also run implementations of classic DQN and a hybrid Evolutionary DQN (ERL-DQN):

### Running DQN

```bash
cd dqn

python train_dqn.py
```
This script will train a Deep Q-Network agent on CartPole-v1. It will display training progress (episode reward and number of steps), and generate a plot:

- A reward curve showing average reward per episode.

### Running Evolutionary DQN

```bash
python train_erl_dqn.py
```
This script trains a population of agents using a hybrid Evolutionary Reinforcement Learning algorithm. During training, it will print:
- Generation index

- Best and average rewards

- Total number of environment steps used

It will also produce: 
- A graph of top agent rewards across generations.

### 7. Troubleshooting

If you get no attribute 'bool8' error, then Downgrade NumPy

```bash
pip install numpy==1.23.5
```

### 8. Authors

- [Chethas Anil Reddy](https://github.com/ChethasReddy)
- [Chaitanya shashi kumar](https://github.com/chaitanya2108)
- [Yejun](https://github.com/madman13131313)

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

### 6. Troubleshooting

If you get no attribute 'bool8' error, then Downgrade NumPy

```bash
pip install numpy==1.23.5
```

### 7. Authors

- [Chethas Anil Reddy](https://github.com/ChethasReddy)
- [Chaitanya shashi kumar](https://github.com/chaitanya2108)
- [Yejun](https://github.com/madman13131313)

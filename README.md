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

### 5. Troubleshooting

If you get no attribute 'bool8' error, then Downgrade NumPy

```bash
pip install numpy==1.23.5
```

### 6. Authors

- [Chethas Anil Reddy](https://github.com/ChethasReddy)
- [Yejun](https://github.com/madman13131313)

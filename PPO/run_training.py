from agent import Agent

def main():
    # Create and train the agent
    agent = Agent()
    print("Starting hybrid Evolutionary-PPO training...")
    agent.train()
    print("Training completed!")

if __name__ == "__main__":
    main()
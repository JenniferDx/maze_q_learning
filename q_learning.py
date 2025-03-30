import pygame
import time
import numpy as np
from maze_env import MazeEnv, Action
import random
import os
import json
import matplotlib.pyplot as plt

def q_learning_agent(log_path, render_path):
    # Create maze environment
    env = MazeEnv()
    
    # Initialize Q-table
    # Q-table shape: (height, width, number of actions)
    q_table = np.zeros((env.height, env.width, len(Action)))
    
    # Q-learning parameters
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 0.1  # Exploration rate
    episodes = 100
    
    # Reset environment
    state = env.reset()
    
    # Learning log to track progress
    learning_log = {
        "episodes": [],
        "steps": [],
        "rewards": [],
        "epsilons": []
    }
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Exploration: choose a random action
                action = random.choice(list(Action))
            else:
                # Get all actions with the maximum Q-value (handle ties)
                q_values = q_table[state[0], state[1]]
                max_q = np.max(q_values)
                max_actions = [a for a, q in enumerate(q_values) if q == max_q]
                # Randomly select one of the best actions
                action = Action(random.choice(max_actions))
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Update Q-table using the Q-learning formula
            old_value = q_table[state[0], state[1], action.value]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            
            # Q-learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state[0], state[1], action.value] = new_value
            
            # Update state
            state = next_state
            step_count += 1
            
        # Log the episode data
        learning_log["episodes"].append(episode + 1)
        learning_log["steps"].append(step_count)
        learning_log["rewards"].append(total_reward)
        learning_log["epsilons"].append(epsilon)
        
        print(f"Episode {episode+1}/{episodes}, Steps: {step_count}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    # Save learning log to JSON file
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, "learning_log_q_learning.json"), "w") as f:
        json.dump(learning_log, f, indent=4)
    
    visualize_learning(learning_log, log_path)

    # Test the trained agent
    test_agent(env, q_table, render_path)
    
    # Wait a bit before closing
    time.sleep(2)
    env.close()

def visualize_learning(learning_log, log_path):
    """Visualize the learning progress"""
    plt.figure(figsize=(15, 10))
    
    # Plot steps per episode
    plt.subplot(3, 1, 1)
    plt.plot(learning_log["episodes"], learning_log["steps"])
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    # Plot rewards per episode
    plt.subplot(3, 1, 2)
    plt.plot(learning_log["episodes"], learning_log["rewards"])
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    # Plot epsilon decay
    plt.subplot(3, 1, 3)
    plt.plot(learning_log["episodes"], learning_log["epsilons"])
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "learning_progress_q_learning.png"))
    #plt.show()

def test_agent(env, q_table, render_path):
    """Test the trained agent using the learned Q-table"""
    print("\nTesting trained agent...")
    state = env.reset()
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 100:  # Limit to 100 steps to avoid infinite loops
        # Choose the best action from Q-table
        action = Action(np.argmax(q_table[state[0], state[1]]))
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Render
        env.render(save_frame=True, save_path=render_path, frame_number=step_count)
        time.sleep(0.2)  # Slow down for visualization
        
        # Update state
        state = next_state
        step_count += 1
        
        # Process pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    print(f"Test finished after {step_count} steps with total reward: {total_reward}")

if __name__ == "__main__":
    log_path = "learning_logs"
    render_path = "rendered_frames/q_learning_demo"
    q_learning_agent(log_path=log_path, render_path=render_path)
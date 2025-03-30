from math import log
import pygame
import time
import numpy as np
from maze_env import MazeEnv, Action
import random
import json
import matplotlib.pyplot as plt
from collections import deque
import os

def q_learning_agent(log_path, render_path, n):
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
    n_steps = n  # Number of steps to look ahead
    
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
        
        # Initialize n-step memory
        n_step_memory = deque(maxlen=n_steps)
        
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
            
            # Store experience in n-step memory
            n_step_memory.append((state, action, reward, next_state, done))
            
            # Only update Q-table if we have enough steps
            if len(n_step_memory) == n_steps:
                # Get the initial state and action
                init_state, init_action, _, _, _ = n_step_memory[0]
                
                # Calculate n-step return
                n_step_return = 0
                for i in range(n_steps):
                    n_step_return += (gamma ** i) * n_step_memory[i][2]  # Discounted reward
                
                # Add bootstrapped value if not done
                if not n_step_memory[-1][4]:  # If the last state is not terminal
                    last_next_state = n_step_memory[-1][3]
                    n_step_return += (gamma ** n_steps) * np.max(q_table[last_next_state[0], last_next_state[1]])
                
                # Update Q-value for the initial state-action pair
                old_value = q_table[init_state[0], init_state[1], init_action.value]
                new_value = old_value + alpha * (n_step_return - old_value)
                q_table[init_state[0], init_state[1], init_action.value] = new_value
            
            # Update state
            state = next_state
            step_count += 1
            
        # Process any remaining experiences in n-step memory
        while n_step_memory:
            # Get the initial state and action
            init_state, init_action, _, _, _ = n_step_memory[0]
            
            # Calculate return for remaining steps
            n_step_return = 0
            for i, (_, _, r, _, _) in enumerate(n_step_memory):
                n_step_return += (gamma ** i) * r
            
            # Add bootstrapped value if not done and there are remaining steps
            if n_step_memory and not n_step_memory[-1][4]:
                last_next_state = n_step_memory[-1][3]
                n_step_return += (gamma ** len(n_step_memory)) * np.max(q_table[last_next_state[0], last_next_state[1]])
            
            # Update Q-value
            old_value = q_table[init_state[0], init_state[1], init_action.value]
            new_value = old_value + alpha * (n_step_return - old_value)
            q_table[init_state[0], init_state[1], init_action.value] = new_value
            
            # Remove the first experience after using it
            n_step_memory.popleft()
        
        # Log the episode data
        learning_log["episodes"].append(episode + 1)
        learning_log["steps"].append(step_count)
        learning_log["rewards"].append(total_reward)
        learning_log["epsilons"].append(epsilon)
        
        print(f"Episode {episode+1}/{episodes}, Steps: {step_count}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    # Save learning log to JSON file
    with open(os.path.join(log_path, f"learning_log_q_learn_{n}_steps.json"), "w") as f:
        json.dump(learning_log, f, indent=4)
    
    # Visualize learning progress
    visualize_learning(learning_log, log_path, n)
    
    # Test the trained agent
    test_agent(env, q_table, render_path)
    
    # Wait a bit before closing
    time.sleep(2)
    env.close()

def visualize_learning(learning_log, log_path, n):
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
    plt.savefig(os.path.join(log_path, f"learning_process_q_learning_{n}_steps.png"))
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
    n = 3
    log_path = "learning_logs"
    render_path = f"rendered_frames/q_learning_{n}_steps"
    q_learning_agent(log_path=log_path, render_path=render_path, n=n)
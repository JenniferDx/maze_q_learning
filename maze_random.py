import pygame
import time
from maze_env import MazeEnv, Action
import random
import os

def random_agent_demo(render_path):
    # Create maze environment
    env = MazeEnv()
    
    # Reset environment
    env.reset()
    
    done = False
    total_reward = 0
    
    # Run until episode is done or max steps reached
    max_steps = 100
    step_count = 0
    
    os.makedirs(render_path, exist_ok=True)
    while not done and step_count < max_steps:
        # Random action
        action = random.choice(list(Action))
        
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
    
    print(f"Episode finished after {step_count} steps with total reward: {total_reward}")
    
    # Wait a bit before closing
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    render_path = "rendered_frames/random_agent_demo"
    random_agent_demo(render_path=render_path)
import numpy as np
import pygame
import os
from enum import Enum

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class MazeEnv:
    def __init__(self):
        """
        Initialize the maze environment
        
        Args:
            width (int): Width of the maze
            height (int): Height of the maze
        """
        self.width = 9
        self.height = 6
        
        # Create the maze
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize maze with zeros (empty cells)
        self.maze = np.zeros((self.height, self.width), dtype=int)
        
        # Set start position
        self.start_pos = (2, 0)
        self.maze[self.start_pos] = 0  # Ensure start position is empty
        
        # Set goal position (bottom-right corner)
        self.goal_pos = (0, self.width - 1)
        self.maze[self.goal_pos] = 0  # Ensure goal position is empty
        
        # Set current position to start position
        self.agent_pos = self.start_pos
        
        # Create some obstacles
        self.maze[1:4, 2] = 1 
        self.maze[4, 5] = 1
        self.maze[0:3, 7] = 1
        
        return self._get_state()
    

    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (Action): The action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get current position
        y, x = self.agent_pos
        
        # Calculate new position based on action
        if action == Action.UP:
            new_pos = (max(0, y - 1), x)
        elif action == Action.RIGHT:
            new_pos = (y, min(self.width - 1, x + 1))
        elif action == Action.DOWN:
            new_pos = (min(self.height - 1, y + 1), x)
        elif action == Action.LEFT:
            new_pos = (y, max(0, x - 1))
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if agent hit a wall (boundary)
        hit_wall = (new_pos[0] == y and new_pos[1] == x and 
                   (action == Action.UP and y == 0 or
                    action == Action.RIGHT and x == self.width - 1 or
                    action == Action.DOWN and y == self.height - 1 or
                    action == Action.LEFT and x == 0))
        
        # Check if new position is valid (not an obstacle)
        if self.maze[new_pos] == 1 or hit_wall:  # Hit an obstacle or wall
            reward = 0
            new_pos = self.agent_pos  # Stay in the same position
            done = False
        else:
            self.agent_pos = new_pos
            
            if new_pos == self.goal_pos:  # Reached the goal
                reward = 1
                done = True
            else:
                reward = 0  # Small penalty for each step to encourage efficiency
                done = False
                
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        """Return the current state representation"""
        return self.agent_pos
    
    def render(self, mode='human', save_frame=False, frame_number=None, save_path=""):
        """Render the environment"""
        if mode == 'human':
            # Initialize pygame if not already done
            if not hasattr(self, 'screen'):
                pygame.init()
                self.cell_size = 50
                self.screen_width = self.width * self.cell_size
                self.screen_height = self.height * self.cell_size
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption('Maze Environment')
                
            # Fill background
            self.screen.fill((255, 255, 255))
            
            # Draw maze
            for y in range(self.height):
                for x in range(self.width):
                    rect = pygame.Rect(
                        x * self.cell_size, 
                        y * self.cell_size, 
                        self.cell_size, 
                        self.cell_size
                    )
                    
                    # Draw cell based on its content
                    if (y, x) == self.start_pos:
                        pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Start: green
                    elif (y, x) == self.goal_pos:
                        pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Goal: red
                    elif (y, x) == self.agent_pos:
                        pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Agent: blue
                    elif self.maze[y, x] == 1:
                        pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Obstacle: black
                    else:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect)  # Empty: white
                    
                    # Draw cell border
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
            
            pygame.display.flip()
            
            # Save the current frame if requested
            if save_frame and frame_number is not None:
                os.makedirs(save_path, exist_ok=True)
                frame_path =  os.path.join(save_path, f"frame_{frame_number:04d}.png")
                pygame.image.save(self.screen, frame_path)

    def close(self):
        """Close the environment"""
        if hasattr(self, 'screen'):
            pygame.quit()

if __name__ == "__main__":
    env = MazeEnv()
    save_path = "rendered_frames"
    os.makedirs(save_path, exist_ok=True)
    env.render(save_frame=True, frame_number=0, save_path=save_path)
    env.close()
    
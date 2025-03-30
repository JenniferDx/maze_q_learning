#draw the json file in learning_logs dir, save the image of eposide vs reward, and episode vs steps
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

def analyze_learning_logs():
    # Directory containing learning logs
    log_dir = "d:/projects/Dyna_maze/learning_logs"
    
    # Create output directory for plots
    output_dir = "d:/projects/Dyna_maze/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the log directory
    json_files = [f for f in os.listdir(log_dir) if f.endswith('.json') or not f.endswith('.png')]
    
    if not json_files:
        print(f"No JSON files found in {log_dir}")
        return
    
    # Prepare data structures for comparison plots
    all_data = {}
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(log_dir, json_file)
        
        # Extract algorithm name and parameters from filename
        algorithm_name = json_file.replace('.json', '')
        
        # Extract n-step value if present
        n_step_match = re.search(r'(\d+)_steps', algorithm_name)
        n_steps = int(n_step_match.group(1)) if n_step_match else 1
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Store data for comparison
            all_data[algorithm_name] = {
                'episodes': data.get('episodes', []),
                'rewards': data.get('rewards', []),
                'steps': data.get('steps', []),
                'n_steps': n_steps
            }
            
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing {json_file}: {e}")
    
    # Create comparison plots
    create_comparison_plots(all_data, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


def create_comparison_plots(all_data, output_dir):
    """Create plots comparing different algorithms"""
    if len(all_data) <= 1:
        return
    
    # Sort algorithms by n_steps value
    sorted_algorithms = sorted(all_data.items(), key=lambda x: x[1]['n_steps'])
    
    # Create comparison plots
    plt.figure(figsize=(15, 5))
    
    
    # Plot episodes vs steps for all algorithms
    for algo_name, algo_data in sorted_algorithms:
        plt.plot(algo_data['episodes'], algo_data['steps'], label=algo_name)
    
    plt.title('Comparison: Episodes vs Steps')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithms_comparison.png"))
    
    # Create a smoothed version for better visualization
    plt.figure(figsize=(15, 5))
    
    # Smoothed steps
    for algo_name, algo_data in sorted_algorithms:
        steps = np.array(algo_data['steps'])
        smoothed_steps = np.convolve(steps, np.ones(5)/5, mode='valid')
        plt.plot(algo_data['episodes'][4:], smoothed_steps, label=algo_name)
    
    plt.title('Comparison: Episodes vs Steps (Smoothed)')
    plt.xlabel('Episodes')
    plt.ylabel('Steps (5-episode moving average)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithms_comparison_smoothed.png"))
    plt.close()

if __name__ == "__main__":
    analyze_learning_logs()
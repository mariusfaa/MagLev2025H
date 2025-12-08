import pandas as pd
import numpy as np
import glob
import os
import argparse

# --- CONFIGURATION ---

# Define the specific order for controllers (Case insensitive matching)
CONTROLLER_ORDER = ['acados', 'standard', 'tube', 'stochastic']

# Define the scenarios to analyze in the specified order
SCENARIOS = [
    "sigmoid_Slope_5",
    "sigmoid_Slope_10",
    "sigmoid_Slope_15",
    "sine_Period_0.1",
    "sine_Period_1",
    "sine_Period_10"
]

# Root directory for results
ROOT_DIR = "benchmark_results"

# --- HELPER FUNCTIONS ---

def get_controller_rank(filename):
    """
    Helper function to determine sort order based on filename.
    Returns the index in CONTROLLER_ORDER or a high number if not found.
    """
    lower_name = filename.lower()
    for index, controller in enumerate(CONTROLLER_ORDER):
        if controller in lower_name:
            return index
    return 999  # Put unknown controllers at the end

def calculate_rmse_for_scenario(base_path, scenario_name):
    """
    Calculates and prints RMSE for a specific scenario folder.
    """
    directory = os.path.join(base_path, scenario_name)
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"\n[!] Folder not found: {scenario_name}")
        return

    # Find all csv files
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"\n[!] No CSV files in: {scenario_name}")
        return

    # Sort files based on the custom controller order
    csv_files.sort(key=lambda x: get_controller_rank(os.path.basename(x)))

    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'Controller / Filename':<40} | {'RMSE':<20}")
    print("-" * 65)

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            # --- IDENTIFY COLUMNS ---
            possible_ref_cols = ['Target', 'reference', 'ref', 'target', 'target_height', 'Target Height', 'r']
            possible_pos_cols = ['Position', 'position', 'pos', 'height', 'ball_height', 'y', 'Height']

            ref_col = next((c for c in df.columns if c in possible_ref_cols), None)
            pos_col = next((c for c in df.columns if c in possible_pos_cols), None)

            if ref_col and pos_col:
                ref_data = df[ref_col].values
                pos_data = df[pos_col].values

                # --- CALCULATE RMSE ---
                mse = np.mean((ref_data - pos_data) ** 2)
                rmse = np.sqrt(mse)

                # Create a cleaner display name if it matches a known controller
                display_name = filename
                for ctrl in CONTROLLER_ORDER:
                    if ctrl in filename.lower():
                        display_name = ctrl.capitalize()
                        break
                
                print(f"{display_name:<40} | {rmse:.5f}")
            
            else:
                print(f"{filename:<40} | ERROR: Columns 'Target'/'Position' not found")

        except Exception as e:
            print(f"{filename:<40} | ERROR: {str(e)}")

def select_timestamp(root_dir):
    """
    Lists available timestamps and asks user to select one.
    Returns the selected timestamp string.
    """
    if not os.path.exists(root_dir):
        print(f"Error: Root directory '{root_dir}' does not exist.")
        return None

    # Get subdirectories (timestamps)
    timestamps = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    if not timestamps:
        print(f"No timestamp folders found in '{root_dir}'.")
        return None

    print(f"\nFound {len(timestamps)} benchmarks in '{root_dir}':")
    print("-" * 40)
    for idx, ts in enumerate(timestamps):
        print(f"{idx + 1:>3}: {ts}")
    print("-" * 40)

    while True:
        user_input = input(f"\nEnter number to select (1-{len(timestamps)}) or press Enter for LATEST ({timestamps[-1]}): ")
        
        # Default to latest if empty
        if user_input.strip() == "":
            return timestamps[-1]
        
        # Try to parse selection
        try:
            selection = int(user_input)
            if 1 <= selection <= len(timestamps):
                return timestamps[selection - 1]
            else:
                print(f"Please enter a number between 1 and {len(timestamps)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    # Setup argument parser for optional command line usage
    parser = argparse.ArgumentParser(description="Analyze Benchmark Results")
    parser.add_argument("--timestamp", type=str, help="Specific timestamp folder to analyze", default=None)
    args = parser.parse_args()

    selected_timestamp = args.timestamp

    # If no timestamp provided via command line, ask interactively
    if selected_timestamp is None:
        selected_timestamp = select_timestamp(ROOT_DIR)
    
    if selected_timestamp:
        full_path = os.path.join(ROOT_DIR, selected_timestamp)
        print(f"\n{'#'*80}")
        print(f"ANALYZING TIMESTAMP: {selected_timestamp}")
        print(f"Full Path: {full_path}")
        print(f"{'#'*80}")

        # Run analysis for each scenario defined in the list
        for scenario in SCENARIOS:
            calculate_rmse_for_scenario(full_path, scenario)
    else:
        print("Analysis aborted.")

if __name__ == "__main__":
    main()
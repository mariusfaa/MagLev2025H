import pstats
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

import pstats
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

class FlexibleProfilerAnalyzer:
    def __init__(self):
        self.color_palette = plt.cm.tab20.colors
        self.method_colors = {}
    
    def extract_method_stats(self, profile_file, methods_to_track):
        """
        Extract average time per call for specified methods from a .prof file
        """
        stats = pstats.Stats(profile_file)
        method_stats = {}
        
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, func_name = func
            
            # Check if this is one of our tracked methods
            for method_name in methods_to_track:
                # Flexible matching: method_name, Class.method, or module.Class.method
                if (method_name == func_name or 
                    f".{method_name}" in func_name or 
                    method_name in func_name):
                    
                    if nc > 0:
                        avg_time_per_call = tt / nc
                        method_stats[method_name] = {
                            'total_time': tt,
                            'cumulative_time': ct,
                            'call_count': nc,
                            'avg_time_per_call': avg_time_per_call
                        }
                    break
        
        # Ensure all requested methods are present (fill with 0 if not called)
        for method_name in methods_to_track:
            if method_name not in method_stats:
                method_stats[method_name] = {
                    'total_time': 0,
                    'cumulative_time': 0,
                    'call_count': 0,
                    'avg_time_per_call': 0
                }
        
        return method_stats
    
    def load_prof_files_with_methods(self, prof_files_methods_dict):
        """
        Load multiple .prof files with their specific methods to track
        
        Args:
            prof_files_methods_dict: Dictionary {prof_file: [methods_to_track]}
        """
        file_data = {}
        all_methods = set()
        
        for prof_file, methods in prof_files_methods_dict.items():
            if Path(prof_file).exists():
                file_data[prof_file] = {
                    'methods_data': self.extract_method_stats(prof_file, methods),
                    'tracked_methods': methods
                }
                all_methods.update(methods)
            else:
                print(f"Warning: File {prof_file} not found")
                file_data[prof_file] = {
                    'methods_data': {method: {
                        'total_time': 0, 'cumulative_time': 0, 
                        'call_count': 0, 'avg_time_per_call': 0
                    } for method in methods},
                    'tracked_methods': methods
                }
                all_methods.update(methods)
        
        return file_data, sorted(list(all_methods))
    
    def create_stacked_plot(self, prof_files_methods_dict, output_file=None, use_total_time=True):
        """
        Create a stacked column plot where each prof file can track different methods
        """
        # Load data from all prof files
        file_data, all_methods = self.load_prof_files_with_methods(prof_files_methods_dict)
        
        # Prepare data for plotting
        files = list(file_data.keys())
        
        # Assign colors to methods
        self._assign_method_colors(all_methods)
        
        # Create stacked data - methods x files
        stack_data = np.zeros((len(all_methods), len(files)))
        
        for j, file_path in enumerate(files):
            methods_data = file_data[file_path]['methods_data']
            tracked_methods = file_data[file_path]['tracked_methods']
            
            for i, method in enumerate(all_methods):
                if method in tracked_methods:  # Only include if method was tracked for this file
                    if use_total_time:
                        stack_data[i, j] = methods_data[method]['total_time']
                    else:
                        stack_data[i, j] = methods_data[method]['avg_time_per_call']
                else:
                    stack_data[i, j] = 0  # Method not tracked in this file
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(8, len(files) * 1.5), 8))
        
        # Create stacked bars
        bottoms = np.zeros(len(files))
        bars = []
        
        for i, method in enumerate(all_methods):
            color = self.method_colors[method]
            bar = ax.bar(range(len(files)), stack_data[i], bottom=bottoms, 
                        label=method, color=color, alpha=0.8,
                        edgecolor='white', linewidth=0.5)
            bars.append(bar)
            bottoms += stack_data[i]
        
        # Customize the plot
        ax.set_xlabel('Profile Runs', fontsize=12)
        if use_total_time:
            ax.set_ylabel('Total Time (seconds)', fontsize=12)
            title_suffix = 'Total Time'
        else:
            ax.set_ylabel('Average Time per Call (seconds)', fontsize=12)
            title_suffix = 'Average Time per Call'
        
        ax.set_title(f'Method Execution Time - {title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(files)))
        
        # Use shortened file names for x-axis labels
        file_labels = [Path(f).stem for f in files]
        ax.set_xticklabels(file_labels, rotation=45, ha='right')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on bars (only for significant segments)
        self._add_value_labels(ax, stack_data, files)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
        
        return file_data, all_methods
    
    def _assign_method_colors(self, all_methods):
        """Assign consistent colors to methods"""
        for i, method in enumerate(all_methods):
            self.method_colors[method] = self.color_palette[i % len(self.color_palette)]
    
    def _add_value_labels(self, ax, stack_data, files):
        """Add value labels to significant bar segments"""
        cumulative_heights = np.zeros(len(files))
        max_height = np.max(np.sum(stack_data, axis=0))
        
        for i in range(len(stack_data)):
            for j, height in enumerate(stack_data[i]):
                if height > max_height * 0.03:  # Only label if segment is >3% of max column
                    label_y = cumulative_heights[j] + height / 2
                    ax.text(j, label_y, f'{height:.3f}s', 
                           ha='center', va='center', fontsize=8, 
                           fontweight='bold', color='black')
            cumulative_heights += stack_data[i]
    
    def print_detailed_summary(self, file_data, all_methods):
        """Print a detailed summary of the collected data"""
        print("\n" + "="*80)
        print("DETAILED PROFILING DATA SUMMARY")
        print("="*80)
        
        for file_path, data in file_data.items():
            methods_data = data['methods_data']
            tracked_methods = data['tracked_methods']
            
            print(f"\nFile: {Path(file_path).name}")
            print("Tracked methods:", ", ".join(tracked_methods))
            print("-" * 60)
            
            total_time = sum(methods_data[method]['total_time'] for method in tracked_methods)
            print(f"Total tracked time: {total_time:.4f}s")
            
            for method in tracked_methods:
                mdata = methods_data[method]
                if mdata['call_count'] > 0:
                    print(f"  {method:20}: {mdata['total_time']:8.4f}s "
                          f"({mdata['call_count']:4d} calls, "
                          f"avg: {mdata['avg_time_per_call']:.6f}s/call)")
                else:
                    print(f"  {method:20}: {mdata['total_time']:8.4f}s (not called)")

# Usage Example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FlexibleProfilerAnalyzer()
    
    # Define which methods to track for each prof file
    prof_files_methods = {
        "EKF_MPC.prof": ["predict", "update"],
        "MHE_MPC.prof": ["set_arrival_cost", "add_measurement", "cost_function", "kalman_update", "solve"],
        "MHE_acados_MPC.prof": ["run_mhe", "kalman_update"]
    }
    
    # Create plot using total time
    #file_data, all_methods = analyzer.create_stacked_plot(
    #    prof_files_methods, 
    #    output_file="flexible_method_times.png",
    #    use_total_time=True
    #)
    #
    ## Print detailed summary
    #analyzer.print_detailed_summary(file_data, all_methods)
    #
    ## Print summary
    #analyzer.print_summary(file_data)
    
    # You can also create a plot using average time per call
    print("\nCreating average time per call plot...")
    analyzer.create_stacked_plot(
        prof_files_methods,
        output_file="method_avg_times_stacked.png", 
        use_total_time=False
    )

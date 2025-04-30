import matplotlib.pyplot as plt
import numpy as np
import os
from visualization import StrategyVisualization, generate_all_visualizations


def generate_sample_data():
    """Generate sample data for testing visualization module
    
    Returns:
        tuple: (scores_list, strategy_scores) containing overall score list and strategy score dictionary
    """
    # Simulate overall score list
    np.random.seed(42)  # Set random seed to ensure reproducible results
    num_samples = 50
    
    # Generate a gradually improving score list with some fluctuations
    base_scores = np.linspace(0.5, 0.9, num_samples)  # Linear growth of base scores
    noise = np.random.normal(0, 0.05, num_samples)    # Add some random fluctuations
    scores_list = base_scores + noise
    
    # Ensure scores are monotonically increasing (take historical maximum)
    for i in range(1, len(scores_list)):
        scores_list[i] = max(scores_list[i], scores_list[i-1])
    
    # Simulate scores for various strategies
    strategy_scores = {
        "hybrid": [],
        "first_fit": [],
        "best_fit": [],
        "worst_fit": [],
        "greedy": [],
        "other": []
    }
    
    # Generate score sequences with different characteristics for each strategy
    # hybrid strategy - best performance, steady improvement
    hybrid_base = np.linspace(0.6, 0.9, num_samples)
    hybrid_noise = np.random.normal(0, 0.03, num_samples)
    strategy_scores["hybrid"] = list(hybrid_base + hybrid_noise)
    
    # first_fit strategy - medium performance with fluctuations
    first_fit_base = np.linspace(0.5, 0.75, num_samples)
    first_fit_noise = np.random.normal(0, 0.05, num_samples)
    strategy_scores["first_fit"] = list(first_fit_base + first_fit_noise)
    
    # best_fit strategy - starts poor but improves later
    best_fit_base = np.linspace(0.4, 0.8, num_samples)
    best_fit_noise = np.random.normal(0, 0.04, num_samples)
    strategy_scores["best_fit"] = list(best_fit_base + best_fit_noise)
    
    # worst_fit strategy - poor performance
    worst_fit_base = np.linspace(0.3, 0.6, num_samples)
    worst_fit_noise = np.random.normal(0, 0.06, num_samples)
    strategy_scores["worst_fit"] = list(worst_fit_base + worst_fit_noise)
    
    # greedy strategy - starts well but stagnates later
    greedy_base = np.concatenate([
        np.linspace(0.55, 0.7, num_samples//2),
        np.linspace(0.7, 0.72, num_samples - num_samples//2)
    ])
    greedy_noise = np.random.normal(0, 0.04, num_samples)
    strategy_scores["greedy"] = list(greedy_base + greedy_noise)
    
    # other strategy - few samples, unstable performance
    other_indices = np.random.choice(range(num_samples), size=num_samples//3, replace=False)
    other_scores = np.random.uniform(0.4, 0.7, size=len(other_indices))
    strategy_scores["other"] = [0] * num_samples
    for idx, score in zip(other_indices, other_scores):
        strategy_scores["other"][idx] = score
    
    # Filter out 0 values (indicating rounds where this strategy wasn't used)
    strategy_scores["other"] = [score for score in strategy_scores["other"] if score > 0]
    
    return list(scores_list), strategy_scores


def demo_individual_visualizations():
    """Demonstrate the use of individual visualization functions"""
    # Generate sample data
    scores_list, strategy_scores = generate_sample_data()
    
    # Create visualization object
    vis = StrategyVisualization(save_dir='visualization_demo')
    
    # 1. Plot overall score progression
    print("\n1. Plotting overall score progression")
    vis.plot_overall_score_progression(scores_list)
    
    # 2. Plot strategy scores progression
    print("\n2. Plotting strategy scores progression")
    vis.plot_strategy_scores(strategy_scores)
    
    # 3. Plot strategy performance comparison (box plot)
    print("\n3. Plotting strategy performance comparison (box plot)")
    vis.plot_strategy_comparison(strategy_scores)
    
    # 4. Plot strategy evolution heatmap
    print("\n4. Plotting strategy evolution heatmap")
    vis.plot_strategy_evolution_heatmap(strategy_scores, window_size=3)
    
    # 5. Plot strategy dominance evolution
    print("\n5. Plotting strategy dominance evolution")
    vis.plot_strategy_dominance(strategy_scores)
    
    # 6. Plot comprehensive dashboard
    print("\n6. Plotting comprehensive dashboard")
    vis.plot_comprehensive_dashboard(scores_list, strategy_scores)


def demo_all_visualizations():
    """Demonstrate generating all visualization charts at once"""
    # Generate sample data
    scores_list, strategy_scores = generate_sample_data()
    
    # Generate all visualization charts
    print("\nGenerating all visualization charts")
    generate_all_visualizations(scores_list, strategy_scores, save_dir='visualization_demo_all')


def demo_real_data_visualization():
    """Try to visualize using real data (if available)"""
    try:
        # Try to import data from funsearch_bin_packing_llm_api.py
        from funsearch_bin_packing_llm_api import scores_list, strategy_scores
        
        # Check if data is valid
        if scores_list and any(scores for scores in strategy_scores.values()):
            print("\nGenerating visualization charts using real data")
            generate_all_visualizations(scores_list, strategy_scores, save_dir='visualization_real_data')
        else:
            print("\nInsufficient real data, unable to generate visualization charts")
            print("Please run funsearch_bin_packing_llm_api.py to generate data first")
    except (ImportError, AttributeError):
        print("\nUnable to import real data, please run funsearch_bin_packing_llm_api.py first")


if __name__ == "__main__":
    print("=== Strategy Visualization Demo Program ===")
    print("This program demonstrates how to use the visualization.py module to generate various visualization charts")
    
    # Create sample data
    print("\nGenerating sample data...")
    sample_scores_list, sample_strategy_scores = generate_sample_data()
    print(f"Generated overall score data for {len(sample_scores_list)} samples")
    for strategy, scores in sample_strategy_scores.items():
        print(f"Generated {len(scores)} score data points for {strategy} strategy")
    
    # Menu selection
    while True:
        print("\nPlease select a demo option:")
        print("1. Demonstrate individual visualization functions")
        print("2. Generate all visualization charts at once (using sample data)")
        print("3. Generate visualization charts using real data (if available)")
        print("0. Exit program")
        
        choice = input("Please enter option number: ")
        
        if choice == '1':
            demo_individual_visualizations()
        elif choice == '2':
            demo_all_visualizations()
        elif choice == '3':
            demo_real_data_visualization()
        elif choice == '0':
            print("Program exited")
            break
        else:
            print("Invalid option, please try again")
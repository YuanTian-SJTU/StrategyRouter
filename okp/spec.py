import numpy as np

def get_items_and_capacity(instance):
    """
    Extract items (weights and values) and knapsack capacity from the instance.
    """
    items = instance['items']  # list of (weight, value) tuples
    capacity = instance['capacity']
    return items, capacity

def calculate_density(items):
    """
    Calculate the value density of items
    items: list of (weight, value) tuples
    """
    return [(value/weight, weight, value) for weight, value in items]

def calculate_total_value(selected_items):
    """
    Calculate the total value of selected items
    selected_items: list of (weight, value) tuples
    """
    return sum(value for _, value in selected_items)

def calculate_total_weight(selected_items):
    """
    Calculate the total weight of selected items
    selected_items: list of (weight, value) tuples
    """
    return sum(weight for weight, _ in selected_items)

def solve_offline_optimal(items, capacity):
    """
    Solve the offline optimal solution using dynamic programming
    """
    n = len(items)
    # dp[i][j] represents the maximum value achievable with first i items and capacity j
    dp = np.zeros((n + 1, capacity + 1))
    
    for i in range(1, n + 1):
        weight, value = items[i-1]
        for j in range(capacity + 1):
            if weight <= j:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight] + value)
            else:
                dp[i][j] = dp[i-1][j]
    
    # Backtrack to find selected items
    selected_items = []
    j = capacity
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i-1][j]:
            selected_items.append(items[i-1])
            j -= items[i-1][0]
    
    return selected_items

def online_knapsack_solver(items, capacity):
    """
    Online knapsack problem solver
    items: list of (weight, value) tuples
    capacity: knapsack capacity
    """
    selected_items = []
    remaining_capacity = capacity
    
    for item in items:
        weight, value = item
        if weight <= remaining_capacity:
            # Use strategy function to decide whether to accept current item
            if accept_item(item, selected_items, remaining_capacity):
                selected_items.append(item)
                remaining_capacity -= weight
    
    return selected_items

@funsearch.run
def evaluate(instances: dict) -> float:
    """
    Evaluate the performance of online knapsack solving strategy
    Calculate the gap between online solution and optimal solution
    """
    total_gap = 0
    solved_instances = 0
    
    for name, instance in instances.items():
        items, capacity = get_items_and_capacity(instance)
        
        # Calculate offline optimal solution
        optimal_items = solve_offline_optimal(items, capacity)
        optimal_value = calculate_total_value(optimal_items)
        
        # Get online algorithm solution
        selected_items = online_knapsack_solver(items, capacity)
        solution_value = calculate_total_value(selected_items)

        if optimal_value:
            gap = ((optimal_value - solution_value) / optimal_value * 100)
            total_gap += gap
            solved_instances += 1
    
    # Return average gap, or a large penalty if no instances were solved
    avg_gap = -total_gap / solved_instances if solved_instances > 0 else -1000
    
    return avg_gap

@funsearch.evolve
def accept_item(current_item: tuple, selected_items: list, remaining_capacity: float) -> bool:
    """
    Adaptive strategy function to decide whether to accept current item
    current_item: (weight, value) tuple
    selected_items: list of already selected items
    remaining_capacity: remaining capacity of knapsack
    """
    weight, value = current_item
    density = value / weight
    
    # Calculate average value density of selected items
    if selected_items:
        avg_density = sum(v/w for w, v in selected_items) / len(selected_items)
    else:
        avg_density = 0
    
    # Accept if current item's density is higher than average
    return density > avg_density

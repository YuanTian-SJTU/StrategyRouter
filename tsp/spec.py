import numpy as np
from tsp.tsp_utils import get_cities_and_dist_matrix

def tsp_distance(path, dist_matrix):
    """Calculate the total distance of the TSP path."""
    return sum(dist_matrix[path[i], path[i+1]] for i in range(len(path)-1)) + dist_matrix[path[-1], path[0]]

def tsp_solver(cities, dist_matrix):
    """Solve the TSP using based on distance priority."""
    n = len(cities)
    unvisited = set(range(n))
    path = [0]  # 从第一个城市出发
    unvisited.remove(0)
    current = 0
    while unvisited:
        priorities = priority(current, list(unvisited), dist_matrix)
        next_city = list(unvisited)[np.argmax(priorities)]
        path.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return path

@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate the average gap between FunSearch and Gurobi solutions."""
    total_gap = 0
    solved_instances = 0
    
    for name, instance in instances.items():
        cities, dist_matrix = get_cities_and_dist_matrix(instance)
        gurobi_distance = instance['Gurobi']
        
        # Get FunSearch solution
        funsearch_path = tsp_solver(cities, dist_matrix)
        funsearch_distance = tsp_distance(funsearch_path, dist_matrix)

        if gurobi_distance:
            gap = ((funsearch_distance - gurobi_distance) / gurobi_distance * 100)
            total_gap += gap
            solved_instances += 1
    
    # Return average gap if any instances were solved, otherwise return a large penalty
    avg_gap = -total_gap / solved_instances if solved_instances > 0 else -1000

    return avg_gap

@funsearch.evolve
def priority(current_city: int, unvisited: list, dist_matrix: np.ndarray) -> np.ndarray:
    """Assign priorities to unvisited cities based on their distances from the current city, the closer the city, the higher the priority."""
    distances = np.array([dist_matrix[current_city, city] for city in unvisited])
    priorities = -distances  # Nearest cities have higher priority
    return priorities

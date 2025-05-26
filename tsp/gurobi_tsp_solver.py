import gurobipy as gp
from gurobipy import GRB

def solve_tsp_gurobi(dist_matrix):
    """
    Solve TSP using Gurobi solver
    """
    n = len(dist_matrix)
    
    # Create model
    model = gp.Model("TSP")
    
    # Create variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
    
    # Set objective
    model.setObjective(gp.quicksum(dist_matrix[i][j] * x[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    
    # Add constraints
    # Each city must be visited exactly once
    for i in range(n):
        model.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1)
        model.addConstr(gp.quicksum(x[j,i] for j in range(n)) == 1)
        # Prevent self-loops
        model.addConstr(x[i,i] == 0)
    
    # Subtour elimination constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1)
    
    # Set time limit to 60 seconds
    model.setParam('TimeLimit', 60)
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.status == GRB.OPTIMAL:
        # Create adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if x[i,j].x > 0.5:
                    adj[i].append(j)
        
        # Construct path
        path = [0]
        current = 0
        while len(path) < n:
            current = adj[current][0]
            path.append(current)
        
        return path, model.objVal
    else:
        print(f"Gurobi solver status: {model.status}")
        return None, None
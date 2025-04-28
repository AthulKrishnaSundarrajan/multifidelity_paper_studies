# Initialize the text file at the start
def initialize_log_file(filename):
    with open(filename, 'w') as f:
        f.write("Optimization Log\n")
        f.write("================\n")
        f.write("Iteration\tObjective Value\tParameters\n")
        f.write("\n")  # Blank line for spacing

# Write iteration results to the text file
def log_iteration(filename, iteration, objective_value, parameters):
    with open(filename, 'a') as f:
        param_str = ", ".join(f"{p:.4f}" for p in parameters)
        f.write(f"{iteration}\t\t{objective_value:.2f}\t\t{param_str}\n")

# Example usage
if __name__ == "__main__":
    filename = "optimization_log.txt"
    
    # Step 1: Initialize file
    initialize_log_file(filename)
    
    # Simulate an optimization process
    for iteration in range(1, 6):
        # (Replace these with real values from your optimization)
        objective_value = 100.0 / iteration
        parameters = [iteration * 0.5, iteration * 1.5]
        
        # Step 2: Log each iteration
        log_iteration(filename, iteration, objective_value, parameters)

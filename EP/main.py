import matplotlib.pyplot as plt
from data_loader import parse_problem_machine_major
from run_ep import run_ep

if __name__ == "__main__":
    problem_data = parse_problem_machine_major()
    if not problem_data:
        print("Exiting due to data loading error.")
        exit()
    P = problem_data["processing_times_matrix"]
    demand_plan = problem_data["demand_plan"]   
    if "1L" not in problem_data["layout_type"]:
        print(f"Warning: This code is for 1-Line (1L) problems.")
        print(f"Data specifies Layout Type: {problem_data['layout_type']}. Results may be incorrect.")
    M = len(P)
    N_types = len(P[0])
    N_total = sum(demand_plan)
    PARENTS_COUNT = 50      
    OFFSPRING_COUNT = 200   
    GENERATIONS = 500
    TOURNAMENT_SIZE = 20    
    USE_LS = False
    print(f"Problem: {M} machines, {N_types} job types, {N_total} total jobs (Mendeley data from STDIN)")
    print(f"Config: pa={PARENTS_COUNT}, os={OFFSPRING_COUNT}, generations={GENERATIONS}, q={TOURNAMENT_SIZE}")
    print(f"Using Local Search (Memetic): {USE_LS}")

    best_perm, best_fitness, history = run_ep(
        processing_times=P,
        demand_plan=demand_plan,
        pa=PARENTS_COUNT,
        os=OFFSPRING_COUNT,
        generations=GENERATIONS,
        tour_size=TOURNAMENT_SIZE,
        use_local_search=USE_LS
    )
    
    print("-" * 50)
    print("COMPLETED!")
    print(f"Best makespan found: {best_fitness}")
    print(f"Best permutation (first 50 jobs): {best_perm[:50]}...")   
    print("\nPlotting convergence curve...")

    plt.plot(history)
    plt.style.use('fivethirtyeight')
    plt.title("EP Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show()
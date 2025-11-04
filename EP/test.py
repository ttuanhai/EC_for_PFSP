import random, math, sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def calculate_makespan(perm, processing_times):
    num_machines = len(processing_times)
    completion_times = [0] * num_machines
    for job in perm:
        for machine in range(num_machines):
            if machine == 0:
                completion_times[machine] += processing_times[machine][job]
            else:
                completion_times[machine] = max(completion_times[machine], completion_times[machine - 1]) + processing_times[machine][job]
    return completion_times[-1]

def insertion_mutation(perm: list) -> list:
    n = len(perm)
    if n < 2:
        return perm[:]
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return perm[:]
    job = perm[i]
    new_perm = perm[:i] + perm[i+1:]
    new_perm.insert(j, job)
    return new_perm

def swap_mutation(perm: list) -> list:
    n = len(perm)
    if n < 2:
        return perm[:]
    i, j = random.sample(range(n), 2)
    new_perm = perm[:]
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm

def displacement_mutation(perm: list, LMax: int) -> list:
    n = len(perm)
    safe_LMax = min(LMax, n // 2)
    if safe_LMax < 2:
        return perm[:]
    L = random.randint(2, safe_LMax)
    i = random.randint(0, n - L)
    new_perm = perm[:]
    block = new_perm[i:i+L]
    del new_perm[i:i+L]
    j = random.randint(0, n - L)
    new_perm[j:j] = block
    return new_perm

def inversion_mutation(perm: list) -> list:
    n = len(perm)
    if n < 2:
        return perm[:]
    i , j = sorted(random.sample(range(n), 2))
    new_perm = perm[:]
    block = new_perm[i:j+1]
    block.reverse()
    new_perm[i:j+1] = block
    return new_perm

def local_search_insert_once(perm, processing_times):
    best_val = calculate_makespan(perm, processing_times)
    best_perm = perm[:]
    for i in range(len(perm)):
        job_to_move = perm[i]
        new_perm = perm[:i] + perm[i+1:]
        for j in range(len(new_perm)+1):
            if i == j:
                continue
            candidate_perm = new_perm[:j] + [job_to_move] + new_perm[j:]
            candidate_val = calculate_makespan(candidate_perm, processing_times)
            if candidate_val < best_val:
                best_val = candidate_val
                best_perm = candidate_perm
    return best_perm

class Individual:
    def __init__(self, perm, p_insert, LMax):
        self.perm = perm
        self.p_insert = p_insert
        self.LMax = LMax
        self.fitness = None

    def mutate_strategy(self):
        p = min(max(self.p_insert, 1e-6), 1 - 1e-6)
        logit_p  = math.log(p / (1 - p))
        logit_p += random.gauss(0, 0.1)
        self.p_insert = 1 / (1 + math.exp(-logit_p))
        
        noise = random.gauss(0, 0.2)
        lmax = self.LMax * math.exp(noise)
        safe_LMax = max(2, len(self.perm) // 2 if len(self.perm) > 3 else 2)
        self.LMax = int(min(max(lmax, 2), safe_LMax))

    def reproduce(self) -> 'Individual':
        new_perm = None
        if random.random() < self.p_insert:
            new_perm = insertion_mutation(self.perm)
        else:
            r = random.random()
            if r < 0.33:
                new_perm = swap_mutation(self.perm)
            elif r < 0.66:
                new_perm = displacement_mutation(self.perm, self.LMax)
            else:
                new_perm = inversion_mutation(self.perm)
        #new_perm = inversion_mutation(self.perm)
                
        child = Individual(new_perm, self.p_insert, self.LMax)
        child.mutate_strategy()
        return child

def parse_problem_machine_major():
    print("Paste your problem (L1, L2, ... L7-end) now.")
    print("On Windows, press Ctrl+Z then Enter. On Mac/Linux, press Ctrl+D.")
    
    try:
        all_text = sys.stdin.read()
        lines = [line.strip() for line in all_text.splitlines() if line.strip()]
        
        if len(lines) < 7:
            print(f"Error: Not enough data pasted. Expected >= 7 lines, got {len(lines)}.")
            return None

        demand_plan_str = lines[0].split()
        demand_plan = [int(p) for p in demand_plan_str]
        
        layout_type = lines[1]
        num_machines_L3 = int(lines[2])
        machines_per_line = lines[3]
        num_machines = int(lines[4]) 
        num_job_types = int(lines[5])

        data_lines = lines[6:]
        
        if len(data_lines) < num_machines:
            print(f"Error: Expected {num_machines} machine rows (from L5), but found {len(data_lines)} data lines.")
            return None
            
        P_machine_major = []
        for i in range(num_machines): 
            parts = data_lines[i].split()
            
            if len(parts) < num_job_types:
                 print(f"Error: Row {i} (Machine {i}) expected {num_job_types} job types (L6), but found {len(parts)}.")
                 return None
                 
            row = [int(parts[j]) for j in range(num_job_types)]
            P_machine_major.append(row)

        if not P_machine_major:
            print("Error: No matrix data was parsed.")
            return None
        
        problem_data = {
            "demand_plan": demand_plan,
            "layout_type": layout_type,
            "num_machines_L3": num_machines_L3,
            "machines_per_line": machines_per_line,
            "num_machines_total_L5": num_machines,
            "num_job_types": num_job_types,
            "processing_times_matrix": P_machine_major
        }
        
        return problem_data

    except Exception as e:
        print(f"Error parsing pasted data: {e}")
        return None

def run_ep(processing_times, demand_plan, pa, os, generations, tour_size, use_local_search=True, run_id=None):
    prefix = f"[RUN {run_id}] " if run_id is not None else ""
    
    population: List[Individual] = []
    
    num_job_types = len(processing_times[0])

    full_job_list = []
    for job_type, num_needed in enumerate(demand_plan):
        if job_type >= num_job_types:
            print(f"Warning: Demand plan has more entries ({len(demand_plan)}) than job types in matrix P ({num_job_types}).")
            break
        full_job_list.extend([job_type] * num_needed)
    
    num_jobs_total = len(full_job_list) 
    if num_jobs_total == 0:
        print("Error: Demand plan is empty. No jobs to schedule.")
        return [], 0, []
        
    safe_LMax_init = max(2, num_jobs_total // 2 if num_jobs_total > 3 else 2)
    
    if run_id == 1:
        print(f"Problem parsed: {len(processing_times)} machines, {num_job_types} job types, {num_jobs_total} total jobs.")

    while len(population) < pa:
        perm = full_job_list[:]
        random.shuffle(perm)
        
        p_insert = random.uniform(0.1, 0.9)
        LMax = random.randint(2, safe_LMax_init)
        ind = Individual(perm, p_insert, LMax)
        population.append(ind)
        
    global_best_perm = None
    global_best_fitness = float('inf')
    
    fitness_history = []
    
    for ind in population:
        if use_local_search:
            ind.perm = local_search_insert_once(ind.perm, processing_times)
        
        ind.fitness = calculate_makespan(ind.perm, processing_times)
        
        if ind.fitness < global_best_fitness:
            global_best_fitness = ind.fitness
            global_best_perm = ind.perm[:]
            
    fitness_history.append(global_best_fitness)
        
    for gen in range(generations):
        offsprings = []
        for _ in range(os):
            parents = random.choice(population)
            child = parents.reproduce()
            
            if use_local_search and random.random() < 0.4:
                child.perm = local_search_insert_once(child.perm, processing_times)
                child.fitness = None
                
            offsprings.append(child)
        
        for child in offsprings:
            if child.fitness is None:
                child.fitness = calculate_makespan(child.perm, processing_times)

        combined = population + offsprings
        wins = [0] * len(combined)
        
        for i in range(len(combined)):
            for _ in range(tour_size):
                j = random.randrange(len(combined))
                if combined[i].fitness < combined[j].fitness:
                    wins[i] += 1
        
        indices = list(range(len(combined)))
        indices.sort(key=lambda x: wins[x], reverse=True)
        
        survivor_indices = indices[:pa]

        population = [combined[i] for i in survivor_indices]

        best_in_gen = population[0]
        
        if best_in_gen.fitness < global_best_fitness:
            global_best_fitness = best_in_gen.fitness
            global_best_perm = best_in_gen.perm[:]
            
        fitness_history.append(global_best_fitness)
    
    return global_best_perm, global_best_fitness, fitness_history

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
    NUM_RUNS = 10
    
    print("--- STARTING EVOLUTIONARY PROGRAMMING (EP) EXPERIMENT ---")
    print(f"Problem: {M} machines, {N_types} job types, {N_total} total jobs")
    print(f"Config: pa={PARENTS_COUNT}, os={OFFSPRING_COUNT}, generations={GENERATIONS}, q={TOURNAMENT_SIZE}")
    print(f"Using Local Search (Memetic): {USE_LS}")
    print(f"Total Runs: {NUM_RUNS}")
    print("-" * 50) 

    all_histories = []
    final_best_fitnesses = []

    for i in range(NUM_RUNS):
        print(f"Starting Run {i + 1}/{NUM_RUNS}...")
        
        best_perm, best_fitness, history = run_ep(
            processing_times=P,
            demand_plan=demand_plan,
            pa=PARENTS_COUNT,
            os=OFFSPRING_COUNT,
            generations=GENERATIONS,
            tour_size=TOURNAMENT_SIZE,
            use_local_search=USE_LS,
            run_id=(i + 1)
        )
        
        all_histories.append(history)
        final_best_fitnesses.append(best_fitness)
        
        print(f"Completed Run {i + 1} | Best Makespan: {best_fitness}")

    print("-" * 50)
    print(f"--- EXPERIMENT COMPLETED ({NUM_RUNS} RUNS) ---")
    
    mean_fitness = np.mean(final_best_fitnesses)
    best_fitness_overall = np.min(final_best_fitnesses)
    worst_fitness = np.max(final_best_fitnesses)
    std_dev = np.std(final_best_fitnesses)

    print(f"Best Makespan Overall: {best_fitness_overall}")
    print(f"Worst Makespan:        {worst_fitness}")
    print(f"Mean Makespan:         {mean_fitness:.2f}")
    print(f"Standard Deviation:    {std_dev:.2f}")
    print("-" * 50)

    print("Plotting results...")
    
    sns.set_theme(style="whitegrid")
    
    np_histories = np.array(all_histories)
    mean_history = np.mean(np_histories, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for history in all_histories:
        ax.plot(history, alpha=0.3, color='grey', linestyle='--')
            
    ax.plot(
        mean_history, 
        color='red',
        linewidth=2,
        label='Average Best Makespan'
    )
            
    ax.set_title("Algorithm Convergence with Inversion Mutation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Makespan")
    ax.grid(True)
    ax.legend()
    
    final_avg_fitness = mean_history[-1]
    
    stats_text = (
        f"Best Makespan: {best_fitness_overall}\n"
        f"Average Makespan: {final_avg_fitness:.2f}"
    )
    
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
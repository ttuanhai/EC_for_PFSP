import random
from individual import Individual
from makespan import calculate_makespan
from local_search import local_search_insert_once

def run_ep(processing_times, demand_plan, pa, os, generations, tour_size, use_local_search=True): 
    population = []   
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
    print(f"Generation 0 | Best Makespan: {global_best_fitness}")

    for gen in range(generations):
        offsprings = []
        for _ in range(os):
            parent = random.choice(population)
            child = parent.reproduce()

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
        if (gen + 1) % 20 == 0:
            print(f"Generation {gen + 1} | Best Makespan: {global_best_fitness}")
    
    return global_best_perm, global_best_fitness, fitness_history
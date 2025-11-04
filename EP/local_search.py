from EP.makespan import calculate_makespan

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
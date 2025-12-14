from makespan import calculate_makespan
def neh_heuristic(full_job_list, processing_times):
    num_types = len(processing_times[0])
    type_total_time = []
    for j in range(num_types):
        total = sum(row[j] for row in processing_times)
        type_total_time.append(total)
    sorted_jobs = sorted(full_job_list, key=lambda type_id: type_total_time[type_id], reverse=True)

    current_seq = [sorted_jobs[0]] 
    
    for i in range(1, len(sorted_jobs)):
        job_to_insert = sorted_jobs[i]
        best_seq = []
        min_makespan = float('inf')
        
        for k in range(len(current_seq) + 1):
            temp_seq = current_seq[:k] + [job_to_insert] + current_seq[k:]
            mk = calculate_makespan(temp_seq, processing_times)
            
            if mk < min_makespan:
                min_makespan = mk
                best_seq = temp_seq
        
        current_seq = best_seq
        
    return current_seq
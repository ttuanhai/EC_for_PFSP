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
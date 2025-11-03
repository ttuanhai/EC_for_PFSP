import sys

def parse_problem_machine_major():
    print("Paste your problem (L1, L2, ... L7-end) now.")
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
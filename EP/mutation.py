import random

def insertion_mutation(perm):
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

def swap_mutation(perm):
    n = len(perm)
    if n < 2:
        return perm[:]
    i, j = random.sample(range(n), 2)
    new_perm = perm[:]
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm

def displacement_mutation(perm, LMax):
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

def inversion_mutation(perm):
    n = len(perm)
    if n < 2:
        return perm[:]
    i , j = sorted(random.sample(range(n), 2))
    new_perm = perm[:]
    block = new_perm[i:j+1]
    block.reverse()
    new_perm[i:j+1] = block
    return new_perm
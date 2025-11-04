import random, math
from mutation import insertion_mutation, swap_mutation, displacement_mutation, inversion_mutation

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

    def reproduce(self):
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
        child = Individual(new_perm, self.p_insert, self.LMax)
        child.mutate_strategy()
        return child
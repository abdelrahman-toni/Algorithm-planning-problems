import numpy as np
import random
import matplotlib.pyplot as plt

random.seed()
np.random.seed()

# ------------------ Problem Parameters ------------------ #
N = 10  # Users/tasks
M = 30  # Edge servers

# User parameters
Pn = [random.randint(500, 2000) for _ in range(N)]  # CPU cycles (million)
qn = [random.randint(100, 512) for _ in range(N)]   # Memory requirement (MB)
Tn = [random.randint(2, 15) * 1000 for _ in range(N)]      # Deadline (msec)
zn = [random.randint(800, 2500) for _ in range(N)]  # Device CPU speed (MHz)
yn = [random.randint(256, 1024) for _ in range(N)]  # Device memory (MB)
dn = [random.uniform(0.05, 0.3) * 1000 for _ in range(N)] # Upload time to cloud (msec)
En = [random.uniform(0.2, 2.0) * 1000 for _ in range(N)]  # Cloud execution time (msec)

# Server parameters
Vm = [random.randint(2000, 10000) for _ in range(M)]  # Server CPU speed (MHz)
Wm = [random.randint(1024, 8192) for _ in range(M)]   # Server memory (MB)

# Transmission times user->server (msec)
tnm = [[random.uniform(0.01, 0.2) * 1000 for _ in range(M)] for _ in range(N)] 

# GA parameters
POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9

CHROM_LENGTH = N * (M + 2)  # Total binary decision variables

fitness_progress = []
# ---------------- GA Functions ---------------- #

def initialize_population():
    population = []
    attempts = 0
    max_attempts = POP_SIZE * 100

    while len(population) < POP_SIZE and attempts < max_attempts:
        chromosome = np.zeros(CHROM_LENGTH, dtype=int)
        for n in range(N):
            assign_idx = random.randint(0, M + 1)
            chromosome[n*(M+2) + assign_idx] = 1

        L, C, S = decode_chromosome(chromosome)
        penalty = check_constraints(L, C, S, Pn, qn, Tn, zn, yn, tnm, Vm, Wm, dn, En)

        if penalty == 0:
            population.append(chromosome)
        attempts += 1

    if len(population) < POP_SIZE:
        print(f"Warning: Only {len(population)} feasible chromosomes generated out of {POP_SIZE}")
    return population

def decode_chromosome(chromosome):
    L = np.zeros(N, dtype=int)
    C = np.zeros(N, dtype=int)
    S = np.zeros((N, M), dtype=int)
    for n in range(N):
        segment = chromosome[n*(M+2):(n+1)*(M+2)]
        L[n] = segment[0]
        C[n] = segment[M+1]
        S[n, :] = segment[1:M+1]
    return L, C, S

def check_task_assignment_constraint(L, C, S):
    for n in range(N):
        if L[n] + C[n] + np.sum(S[n, :]) != 1:
            return False
    return True

def compute_completion_time(L, C, S, p, z, t, V, d, E):
    task_times = np.zeros(N)
    server_load = np.zeros(M)

    for n in range(N):
        for m in range(M):
            if S[n, m] == 1:
                server_load[m] += p[n]

    for n in range(N):
        if L[n] == 1:
            task_times[n] = p[n] / z[n]
        elif C[n] == 1:
            task_times[n] = 2 * d[n] + E[n]
        else:
            for m in range(M):
                if S[n, m] == 1:
                    task_times[n] = 2 * t[n][m] + server_load[m] / V[m]
                    break
    return np.mean(task_times)

def check_constraints(L, C, S, p, q, T, z, y, t, V, W, d, E):
    penalty = 0
    server_memory = np.zeros(M)

    for n in range(N):
        if L[n] == 1 and q[n] > y[n]:
            penalty += 1e6
        for m in range(M):
            if S[n, m] == 1:
                server_memory[m] += q[n]

    for m in range(M):
        if server_memory[m] > W[m]:
            penalty += 1e6 * (server_memory[m] - W[m])

    for n in range(N):
        time = 0
        if L[n] == 1:
            time = p[n] / z[n]
        elif C[n] == 1:
            time = 2 * d[n] + E[n]
        else:
            for m in range(M):
                if S[n, m] == 1:
                    server_load = sum(p[i] for i in range(N) if S[i, m] == 1)
                    time = 2 * t[n][m] + server_load / V[m]
                    break
        if time > T[n]:
            penalty += 1e6 * (time - T[n])

    if not check_task_assignment_constraint(L, C, S):
        penalty += 1e7

    return penalty

def fitness(chromosome, p, q, T, z, y, t, V, W, d, E):
    L, C, S = decode_chromosome(chromosome)
    comp_time = compute_completion_time(L, C, S, p, z, t, V, d, E)
    penalty = check_constraints(L, C, S, p, q, T, z, y, t, V, W, d, E)
    return comp_time + penalty

def select(population, fitnesses):
    selected = []
    for _ in range(2):
        aspirants_idx = random.sample(range(len(population)), 3)
        best_idx = min(aspirants_idx, key=lambda i: fitnesses[i])
        selected.append(population[best_idx])
    return selected

def crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, CHROM_LENGTH - 2)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(chromosome, mutation_prob):
    chromosome = chromosome.copy()
    for idx in range(len(chromosome)):
        if random.random() < mutation_prob:
            chromosome[idx] = 1 - chromosome[idx]

    for n in range(N):
        segment = chromosome[n*(M+2):(n+1)*(M+2)]
        ones = np.where(segment == 1)[0]
        if len(ones) == 0:
            pos = random.randint(0, M+1)
            chromosome[n*(M+2) + pos] = 1
        elif len(ones) > 1:
            keep = random.choice(ones)
            for pos in ones:
                if pos != keep:
                    chromosome[n*(M+2) + pos] = 0
    return chromosome

def genetic_algorithm(N, M, p, q, T, z, y, d, E, t, V, W,
                      pop_size, generations,
                      mutation_prob, crossover_prob):

    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitnesses = [fitness(ind, p, q, T, z, y, t, V, W, d, E) for ind in population]
        for i in range(pop_size):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_solution = population[i].copy()

        offspring = []
        while len(offspring) < pop_size:
            parents = select(population, fitnesses)
            child1, child2 = crossover(parents[0], parents[1], crossover_prob)
            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)
            offspring.extend([child1, child2])

        combined_population = population + offspring
        combined_fitnesses = [fitness(ind, p, q, T, z, y, t, V, W, d, E) for ind in combined_population]
        sorted_indices = np.argsort(combined_fitnesses)
        population = [combined_population[i] for i in sorted_indices[:pop_size]]
        
        fitness_progress.append(best_fitness)

        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen} | Best Fitness: {best_fitness:.4f}")

    return best_solution, best_fitness, fitness_progress

# ---------------- Run GA ---------------- #

if __name__ == "__main__":
    best_sol, best_fit,fitness_progress = genetic_algorithm(
        N=N, M=M,
        p=Pn, q=qn, T=Tn, z=zn, y=yn, d=dn, E=En, t=tnm, V=Vm, W=Wm,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_prob=MUTATION_RATE,
        crossover_prob=CROSSOVER_RATE
    )

    L, C, S = decode_chromosome(best_sol)
    print("\nBest assignment summary:")
    for n in range(N):
        if L[n] == 1:
            print(f"Task {n} assigned LOCAL")
        elif C[n] == 1:
            print(f"Task {n} assigned CLOUD")
        else:
            m = np.where(S[n, :] == 1)[0][0]
            m = m + 1
            print(f"Task {n} assigned SERVER {m}")
    print(f"\nBest fitness (avg completion time + penalties): {best_fit:.4f} ms")
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_progress, label="Best Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.title("Fitness Progress Over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
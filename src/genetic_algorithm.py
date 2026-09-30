"""
genetic_algorithm.py — Real-valued GA for evolving offensive football strategies.

BIOLOGY → GA ANALOGY
=====================
Organism    : One complete offensive strategy — a play-calling profile that
              determines how an offense behaves on every scrimmage play.

Genome      : A real-valued 4-element NumPy vector
              [run_prob, short_pass_prob, deep_pass_prob, fourth_down_aggression]
              Analogous to the full DNA sequence of one organism.

Gene        : One of the 4 scalar parameters (e.g., run_prob).
              Analogous to a single gene locus on a chromosome.

Allele      : The numeric value at a gene (e.g., run_prob = 0.45).
              Analogous to a specific variant of a gene.

Constraint  : run_prob + short_pass_prob + deep_pass_prob == 1.0 (always normalized).
              Like a developmental constraint — the genome must encode a valid organism.

Fitness     : Mean points scored per drive across 100 simulated drives.
              Analogous to reproductive success — higher-scoring strategies
              are more likely to pass their genes to the next generation.

Selection   : Tournament selection (k=5): 5 random individuals compete; the best
              is chosen as a parent. Mimics natural selection — fitter organisms
              are more likely to reproduce, but randomness preserves diversity.

Crossover   : Uniform crossover — each gene is independently inherited from
              parent A or B with equal probability (50/50 per gene). Analogous to
              sexual reproduction and independent assortment of chromosomes.
              After crossover, play probs are renormalized so they sum to 1.0.

Mutation    : Gaussian noise (μ=0, σ=0.05) is added to each gene independently.
              Values are clipped to [0, 1] and renormalized. Analogous to point
              mutations introducing small random variation in offspring.

Elitism     : Top 5 individuals survive unchanged into the next generation.
              Analogous to highly fit organisms that reliably reproduce and whose
              offspring inherit their superior traits with fidelity.

Generation  : One full cycle of evaluation → selection → crossover → mutation.
              Analogous to one breeding cycle in a population.

Population  : 100 strategies evolving simultaneously under shared selection pressure.
              Analogous to a population of organisms in a shared environment.
"""

import numpy as np

# Gene indices for clarity
IDX_RUN   = 0
IDX_SHORT = 1
IDX_DEEP  = 2
IDX_4TH   = 3

GENOME_SIZE = 4


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _normalize(genome: np.ndarray) -> np.ndarray:
    """
    Clip all values to [0, 1], then renormalize play probs so they sum to 1.0.
    fourth_down_aggression is only clipped.
    """
    g = genome.copy().astype(float)
    g = np.clip(g, 0.0, 1.0)

    total = g[:3].sum()
    if total > 1e-9:
        g[:3] /= total
    else:
        g[:3] = np.array([1/3, 1/3, 1/3])

    return g


# ---------------------------------------------------------------------------
# Individual & population creation
# ---------------------------------------------------------------------------
def random_individual() -> np.ndarray:
    """Sample a random genome uniformly over the valid parameter space."""
    play_probs = np.random.uniform(0.0, 1.0, 3)
    play_probs /= play_probs.sum()          # normalize to simplex
    fourth = np.random.uniform(0.0, 1.0)
    return np.array([play_probs[0], play_probs[1], play_probs[2], fourth])


def random_population(size: int) -> list:
    return [random_individual() for _ in range(size)]


# ---------------------------------------------------------------------------
# Genome ↔ Strategy conversion
# ---------------------------------------------------------------------------
def genome_to_strategy(genome: np.ndarray):
    """Convert a genome vector into a simulator.Strategy object."""
    from simulator import Strategy
    return Strategy(
        run_prob=float(genome[IDX_RUN]),
        short_pass_prob=float(genome[IDX_SHORT]),
        deep_pass_prob=float(genome[IDX_DEEP]),
        fourth_down_aggression=float(genome[IDX_4TH]),
    )


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------
def evaluate_population(population: list, n_drives: int = 100,
                         probs: dict = None,
                         constraints: bool = True,
                         variance_penalty: bool = True,
                         turnover_penalty: float = 0.0,
                         field_position_epa: bool = False) -> list:
    """
    Evaluate fitness for every individual in the population.

    Returns a list of floats (fitness per individual).
    probs: pre-loaded probability dict — pass this in so we don't reload
           from disk on every call.
    """
    from simulator import evaluate_strategy
    return [
        evaluate_strategy(genome_to_strategy(g), n_drives=n_drives,
                          verbose=False, probs=probs,
                          constraints=constraints,
                          variance_penalty=variance_penalty,
                          turnover_penalty=turnover_penalty,
                          field_position_epa=field_position_epa)
        for g in population
    ]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def tournament_select(population: list, fitnesses: list, k: int = 5) -> np.ndarray:
    """
    Tournament selection: draw k individuals at random, return the genome
    of the one with the highest fitness. No replacement within a tournament.
    """
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx].copy()


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------
def uniform_crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
    """
    Uniform crossover: each gene is independently drawn from parent A or B
    with equal probability (50 / 50). The resulting child genome is
    renormalized so play probs sum to 1.0.
    """
    mask = np.random.random(GENOME_SIZE) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return _normalize(child)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------
def mutate(genome: np.ndarray, std: float = 0.05) -> np.ndarray:
    """
    Gaussian mutation: add independent N(0, std) noise to every gene.
    Clip all values to [0, 1] and renormalize play probs afterward.
    """
    g = genome + np.random.normal(0.0, std, GENOME_SIZE)
    return _normalize(g)


# ---------------------------------------------------------------------------
# One generation of evolution
# ---------------------------------------------------------------------------
def evolve_generation(population: list, fitnesses: list,
                       elite_n: int = 5,
                       tournament_k: int = 5,
                       mutation_std: float = 0.05) -> list:
    """
    Produce the next generation:

    1. Elitism  — copy the top `elite_n` individuals unchanged.
    2. Offspring — fill the rest with tournament selection → uniform
                   crossover → Gaussian mutation.

    Parameters
    ----------
    population   : list of genome arrays (current generation)
    fitnesses    : corresponding fitness values
    elite_n      : number of top individuals preserved unchanged
    tournament_k : tournament size for parent selection
    mutation_std : std dev of Gaussian mutation noise

    Returns
    -------
    list of genome arrays (next generation, same size as input)
    """
    pop_size = len(population)
    sorted_idx = np.argsort(fitnesses)[::-1]   # descending by fitness

    new_pop: list = []

    # ---- 1. Elitism ----
    for i in range(elite_n):
        new_pop.append(population[sorted_idx[i]].copy())

    # ---- 2. Offspring ----
    while len(new_pop) < pop_size:
        parent_a = tournament_select(population, fitnesses, k=tournament_k)
        parent_b = tournament_select(population, fitnesses, k=tournament_k)
        child = uniform_crossover(parent_a, parent_b)
        child = mutate(child, std=mutation_std)
        new_pop.append(child)

    return new_pop

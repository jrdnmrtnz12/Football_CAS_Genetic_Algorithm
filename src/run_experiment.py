"""
run_experiment.py — Run the full GA experiment, save results and figures.

Usage:
    python src/run_experiment.py
"""

import contextlib
import io
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src/ is on the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from genetic_algorithm import (
    random_population,
    evaluate_population,
    evolve_generation,
    genome_to_strategy,
)
from load_data import load_probabilities
from simulator import simulate_drive

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
POP_SIZE      = 100
N_GENERATIONS = 100
N_DRIVES      = 100     # drives per fitness evaluation
ELITE_N       = 5
TOURNAMENT_K  = 5
MUTATION_STD  = 0.05

RUN_SUBDIR  = 'run3'
RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', RUN_SUBDIR))
FIGURES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'figures', RUN_SUBDIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _param_stats(population: list) -> dict:
    """Compute mean and std of each genome parameter across the population."""
    arr = np.array(population)
    return {
        'mean_run':   arr[:, 0].mean(), 'std_run':   arr[:, 0].std(),
        'mean_short': arr[:, 1].mean(), 'std_short': arr[:, 1].std(),
        'mean_deep':  arr[:, 2].mean(), 'std_deep':  arr[:, 2].std(),
        'mean_4th':   arr[:, 3].mean(), 'std_4th':   arr[:, 3].std(),
    }


def _elapsed(start: float) -> str:
    s = time.time() - start
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------
def run_ga(seed: int = None):
    # If no seed supplied, draw one from OS entropy so the run is
    # reproducible if logged, but not deterministically repeated.
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    print(f"  Experiment seed: {seed}  ← log this to reproduce\n")

    print("Loading probability distributions from disk...")
    probs = load_probabilities()
    print("  Done.\n")

    print(f"Initialising population  (size={POP_SIZE}, seed={seed})")
    population = random_population(POP_SIZE)

    # History containers
    generations   = []
    best_fitness  = []
    mean_fitness  = []
    worst_fitness = []
    mean_run      = [];  std_run   = []
    mean_short    = [];  std_short = []
    mean_deep     = [];  std_deep  = []
    mean_4th      = [];  std_4th   = []

    start = time.time()
    print(f"\n{'Gen':>4}  {'Best':>6}  {'Mean':>6}  {'Worst':>6}  "
          f"{'run':>5}  {'short':>5}  {'deep':>5}  {'4th':>5}  elapsed")
    print("-" * 72)

    for gen in range(N_GENERATIONS):
        # ---- Evaluate ----
        fitnesses = evaluate_population(population, n_drives=N_DRIVES, probs=probs)
        stats     = _param_stats(population)

        # ---- Record ----
        generations.append(gen)
        best_fitness.append(max(fitnesses))
        mean_fitness.append(float(np.mean(fitnesses)))
        worst_fitness.append(min(fitnesses))
        mean_run.append(stats['mean_run']);   std_run.append(stats['std_run'])
        mean_short.append(stats['mean_short']); std_short.append(stats['std_short'])
        mean_deep.append(stats['mean_deep']);  std_deep.append(stats['std_deep'])
        mean_4th.append(stats['mean_4th']);   std_4th.append(stats['std_4th'])

        print(f"{gen:>4}  {max(fitnesses):>6.3f}  {np.mean(fitnesses):>6.3f}  "
              f"{min(fitnesses):>6.3f}  "
              f"{stats['mean_run']:>5.3f}  {stats['mean_short']:>5.3f}  "
              f"{stats['mean_deep']:>5.3f}  {stats['mean_4th']:>5.3f}  "
              f"{_elapsed(start)}")
        sys.stdout.flush()

        # ---- Evolve ----
        population = evolve_generation(
            population, fitnesses,
            elite_n=ELITE_N,
            tournament_k=TOURNAMENT_K,
            mutation_std=MUTATION_STD,
        )

    # ---- Final evaluation (generation N_GENERATIONS) ----
    fitnesses = evaluate_population(population, n_drives=N_DRIVES, probs=probs)
    stats     = _param_stats(population)

    generations.append(N_GENERATIONS)
    best_fitness.append(max(fitnesses))
    mean_fitness.append(float(np.mean(fitnesses)))
    worst_fitness.append(min(fitnesses))
    mean_run.append(stats['mean_run']);   std_run.append(stats['std_run'])
    mean_short.append(stats['mean_short']); std_short.append(stats['std_short'])
    mean_deep.append(stats['mean_deep']);  std_deep.append(stats['std_deep'])
    mean_4th.append(stats['mean_4th']);   std_4th.append(stats['std_4th'])

    print(f"{N_GENERATIONS:>4}  {max(fitnesses):>6.3f}  {np.mean(fitnesses):>6.3f}  "
          f"{min(fitnesses):>6.3f}  "
          f"{stats['mean_run']:>5.3f}  {stats['mean_short']:>5.3f}  "
          f"{stats['mean_deep']:>5.3f}  {stats['mean_4th']:>5.3f}  "
          f"{_elapsed(start)}")

    total_time = time.time() - start
    print(f"\nTotal runtime: {total_time/60:.1f} min")

    # Best individual
    best_idx    = int(np.argmax(fitnesses))
    best_genome = population[best_idx]
    best_strat  = genome_to_strategy(best_genome)

    print(f"\nBest evolved strategy: {best_strat}")
    print(f"  Fitness: {max(fitnesses):.4f} pts/drive")

    # ---- Package history ----
    history = {
        'generation':   generations,
        'best_fitness': best_fitness,
        'mean_fitness': mean_fitness,
        'worst_fitness': worst_fitness,
        'mean_run':     mean_run,   'std_run':   std_run,
        'mean_short':   mean_short, 'std_short': std_short,
        'mean_deep':    mean_deep,  'std_deep':  std_deep,
        'mean_4th':     mean_4th,   'std_4th':   std_4th,
    }

    return history, best_genome, population, fitnesses, probs


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
def save_results(history: dict, best_genome: np.ndarray,
                 population: list, fitnesses: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fitness history
    df_fitness = pd.DataFrame({
        'generation':    history['generation'],
        'best_fitness':  history['best_fitness'],
        'mean_fitness':  history['mean_fitness'],
        'worst_fitness': history['worst_fitness'],
    })
    path = os.path.join(RESULTS_DIR, 'fitness_history.csv')
    df_fitness.to_csv(path, index=False)
    print(f"  Saved {path}")

    # Parameter evolution
    df_params = pd.DataFrame({
        'generation':  history['generation'],
        'mean_run':    history['mean_run'],   'std_run':   history['std_run'],
        'mean_short':  history['mean_short'], 'std_short': history['std_short'],
        'mean_deep':   history['mean_deep'],  'std_deep':  history['std_deep'],
        'mean_4th':    history['mean_4th'],   'std_4th':   history['std_4th'],
    })
    path = os.path.join(RESULTS_DIR, 'parameter_evolution.csv')
    df_params.to_csv(path, index=False)
    print(f"  Saved {path}")

    # Final population with fitnesses
    arr = np.array(population)
    df_pop = pd.DataFrame({
        'run_prob':               arr[:, 0],
        'short_pass_prob':        arr[:, 1],
        'deep_pass_prob':         arr[:, 2],
        'fourth_down_aggression': arr[:, 3],
        'fitness':                fitnesses,
    }).sort_values('fitness', ascending=False).reset_index(drop=True)
    path = os.path.join(RESULTS_DIR, 'final_population.csv')
    df_pop.to_csv(path, index=False)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(history: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    gens = history['generation']

    # ---- Figure 1: Fitness over generations ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, history['best_fitness'],  color='#2ecc71', lw=2.5, label='Best')
    ax.plot(gens, history['mean_fitness'],  color='#3498db', lw=2.5, label='Mean')
    ax.plot(gens, history['worst_fitness'], color='#e74c3c', lw=2.0,
            linestyle='--', label='Worst')
    ax.fill_between(gens, history['worst_fitness'], history['best_fitness'],
                    alpha=0.10, color='#3498db')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Fitness (avg pts / drive)', fontsize=13)
    ax.set_title('Fitness Over Generations — CAS Football GA', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fitness_over_generations.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ---- Figure 2: Strategy parameter evolution ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, history['mean_run'],   color='#e67e22', lw=2.5, label='run_prob')
    ax.plot(gens, history['mean_short'], color='#3498db', lw=2.5, label='short_pass_prob')
    ax.plot(gens, history['mean_deep'],  color='#9b59b6', lw=2.5, label='deep_pass_prob')
    ax.plot(gens, history['mean_4th'],   color='#2ecc71', lw=2.0,
            linestyle='--', label='fourth_down_aggression')
    ax.axhline(1/3, color='gray', lw=1.0, linestyle=':', alpha=0.5,
               label='Equal play probs (1/3)')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Mean parameter value', fontsize=13)
    ax.set_title('Strategy Parameter Evolution — Population Means', fontsize=15)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'strategy_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ---- Figure 3: Population diversity ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, history['std_run'],   color='#e67e22', lw=2.5, label='run_prob')
    ax.plot(gens, history['std_short'], color='#3498db', lw=2.5, label='short_pass_prob')
    ax.plot(gens, history['std_deep'],  color='#9b59b6', lw=2.5, label='deep_pass_prob')
    ax.plot(gens, history['std_4th'],   color='#2ecc71', lw=2.0,
            linestyle='--', label='fourth_down_aggression')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Std dev of parameter across population', fontsize=13)
    ax.set_title('Population Diversity Over Generations', fontsize=15)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'population_diversity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Best strategy — play-by-play log
# ---------------------------------------------------------------------------
def save_best_strategy_drives(best_genome: np.ndarray, probs: dict, n: int = 10):
    """Run the best strategy for n drives (verbose) and save to results/."""
    strat = genome_to_strategy(best_genome)
    np.random.seed(7)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Best Evolved Strategy: {strat}\n")
        total = 0
        for i in range(1, n + 1):
            pts = simulate_drive(strat, drive_num=i, verbose=True, probs=probs)
            total += pts
        print(f"\n{'='*55}")
        print(f"  {n}-drive total: {total} pts  |  avg: {total/n:.2f} pts/drive")
        print(f"{'='*55}")

    text = buf.getvalue()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'best_strategy_drives.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f"  Saved {path}")

    # Also print to terminal so ANSI colors render
    print(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 72)
    print(f"  CAS Football GA — run_experiment.py  [{RUN_SUBDIR}]")
    print(f"  Pop={POP_SIZE}  Gens={N_GENERATIONS}  Drives/eval={N_DRIVES}  "
          f"Elite={ELITE_N}  k={TOURNAMENT_K}  σ={MUTATION_STD}")
    print(f"  Constraints: short-yardage(≤3→deep≤20%)  "
          f"own-territory(<20→deep≤30%)  long-down(>15→short+0.20)")
    print(f"  Fitness:     mean(pts) − 0.5·std(pts)")
    print("=" * 72, "\n")

    history, best_genome, final_pop, final_fits, probs = run_ga()

    print("\n--- Saving results ---")
    save_results(history, best_genome, final_pop, final_fits)

    print("\n--- Generating figures ---")
    make_figures(history)

    print("\n--- Running best strategy (10 verbose drives) ---")
    save_best_strategy_drives(best_genome, probs)

    # ------------------------------------------------------------------
    # Cross-run comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  CROSS-RUN COMPARISON")
    print("=" * 72)
    _root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    _hdr  = f"  {'Run':<8}  {'Best (final gen)':>17}  {'Mean (final gen)':>17}  {'Worst (final gen)':>18}"
    print(_hdr)
    print("  " + "-" * 65)
    for _run in ('run2', 'run3'):
        _csv = os.path.join(_root, _run, 'fitness_history.csv')
        if os.path.exists(_csv):
            _df  = pd.read_csv(_csv)
            _last = _df.iloc[-1]
            print(f"  {_run:<8}  {_last['best_fitness']:>17.4f}  "
                  f"{_last['mean_fitness']:>17.4f}  {_last['worst_fitness']:>18.4f}")
        else:
            print(f"  {_run:<8}  (no data)")
    print("=" * 72)
    print("\nDone.")

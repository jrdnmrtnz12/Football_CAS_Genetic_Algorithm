"""
run_run4.py — Runs 4a and 4b (turnover-penalty experiments).

Run 4a: f = mean(pts)               turnover_penalty=-3.5  constraints=False
Run 4b: f = mean(pts) - 0.5*std     turnover_penalty=-3.5  constraints=False

Also:
  - Generates missing strategy_parameter_evolution.png for run1/run2/run3
    from their saved parameter_evolution.csv files.
  - Saves seed metadata for each new run.
  - Prints the full cross-run comparison table.
"""

import contextlib
import io
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from genetic_algorithm import (
    random_population, evaluate_population, evolve_generation, genome_to_strategy,
)
from load_data import load_probabilities
from simulator import simulate_drive

# ---------------------------------------------------------------------------
# Shared GA config
# ---------------------------------------------------------------------------
POP_SIZE     = 100
N_GEN        = 100
N_DRIVES     = 100
ELITE_N      = 5
TOURNAMENT_K = 5
MUT_STD      = 0.05
TO_PENALTY   = -3.5   # EPA-based turnover cost (Run 4a and 4b)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def _results_dir(run): return os.path.join(_BASE, 'results', run)
def _figures_dir(run): return os.path.join(_BASE, 'figures', run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _param_stats(population):
    arr = np.array(population)
    return {
        'mean_run':   arr[:, 0].mean(), 'std_run':   arr[:, 0].std(),
        'mean_short': arr[:, 1].mean(), 'std_short': arr[:, 1].std(),
        'mean_deep':  arr[:, 2].mean(), 'std_deep':  arr[:, 2].std(),
        'mean_4th':   arr[:, 3].mean(), 'std_4th':   arr[:, 3].std(),
    }


def _elapsed(start):
    s = time.time() - start
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# GA loop (parameterised for 4a / 4b)
# ---------------------------------------------------------------------------
def run_ga(run_label, variance_penalty, constraints, turnover_penalty, probs):
    seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    print(f"  Experiment seed: {seed}  ← log this to reproduce\n")

    # persist seed
    rdir = _results_dir(run_label)
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, 'seed.txt'), 'w') as f:
        f.write(str(seed) + '\n')

    print(f"Initialising population  (size={POP_SIZE}, seed={seed})")
    population = random_population(POP_SIZE)

    generations   = []
    best_fitness  = []; mean_fitness  = []; worst_fitness = []
    mean_run = []; std_run = []; mean_short = []; std_short = []
    mean_deep = []; std_deep = []; mean_4th = []; std_4th = []

    start = time.time()
    print(f"\n{'Gen':>4}  {'Best':>7}  {'Mean':>7}  {'Worst':>7}  "
          f"{'run':>5}  {'short':>5}  {'deep':>5}  {'4th':>5}  elapsed")
    print("-" * 75)

    for gen in range(N_GEN):
        fitnesses = evaluate_population(
            population, n_drives=N_DRIVES, probs=probs,
            constraints=constraints, variance_penalty=variance_penalty,
            turnover_penalty=turnover_penalty,
        )
        stats = _param_stats(population)

        generations.append(gen)
        best_fitness.append(max(fitnesses));  mean_fitness.append(float(np.mean(fitnesses)))
        worst_fitness.append(min(fitnesses))
        mean_run.append(stats['mean_run']);   std_run.append(stats['std_run'])
        mean_short.append(stats['mean_short']); std_short.append(stats['std_short'])
        mean_deep.append(stats['mean_deep']);  std_deep.append(stats['std_deep'])
        mean_4th.append(stats['mean_4th']);   std_4th.append(stats['std_4th'])

        print(f"{gen:>4}  {max(fitnesses):>7.3f}  {np.mean(fitnesses):>7.3f}  "
              f"{min(fitnesses):>7.3f}  "
              f"{stats['mean_run']:>5.3f}  {stats['mean_short']:>5.3f}  "
              f"{stats['mean_deep']:>5.3f}  {stats['mean_4th']:>5.3f}  "
              f"{_elapsed(start)}")
        sys.stdout.flush()

        population = evolve_generation(
            population, fitnesses,
            elite_n=ELITE_N, tournament_k=TOURNAMENT_K, mutation_std=MUT_STD,
        )

    # final eval
    fitnesses = evaluate_population(
        population, n_drives=N_DRIVES, probs=probs,
        constraints=constraints, variance_penalty=variance_penalty,
        turnover_penalty=turnover_penalty,
    )
    stats = _param_stats(population)

    generations.append(N_GEN)
    best_fitness.append(max(fitnesses));  mean_fitness.append(float(np.mean(fitnesses)))
    worst_fitness.append(min(fitnesses))
    mean_run.append(stats['mean_run']);   std_run.append(stats['std_run'])
    mean_short.append(stats['mean_short']); std_short.append(stats['std_short'])
    mean_deep.append(stats['mean_deep']);  std_deep.append(stats['std_deep'])
    mean_4th.append(stats['mean_4th']);   std_4th.append(stats['std_4th'])

    print(f"{N_GEN:>4}  {max(fitnesses):>7.3f}  {np.mean(fitnesses):>7.3f}  "
          f"{min(fitnesses):>7.3f}  "
          f"{stats['mean_run']:>5.3f}  {stats['mean_short']:>5.3f}  "
          f"{stats['mean_deep']:>5.3f}  {stats['mean_4th']:>5.3f}  "
          f"{_elapsed(start)}")
    print(f"\nTotal runtime: {(time.time()-start)/60:.1f} min")

    best_idx    = int(np.argmax(fitnesses))
    best_genome = population[best_idx]
    print(f"\nBest evolved strategy: {genome_to_strategy(best_genome)}")
    print(f"  Fitness: {max(fitnesses):.4f}")

    history = {
        'generation': generations,
        'best_fitness': best_fitness, 'mean_fitness': mean_fitness,
        'worst_fitness': worst_fitness,
        'mean_run': mean_run, 'std_run': std_run,
        'mean_short': mean_short, 'std_short': std_short,
        'mean_deep': mean_deep, 'std_deep': std_deep,
        'mean_4th': mean_4th, 'std_4th': std_4th,
    }
    return history, best_genome, population, fitnesses, seed


# ---------------------------------------------------------------------------
# Save results CSVs
# ---------------------------------------------------------------------------
def save_results(run_label, history, best_genome, population, fitnesses):
    rdir = _results_dir(run_label)
    os.makedirs(rdir, exist_ok=True)

    pd.DataFrame({
        'generation': history['generation'],
        'best_fitness': history['best_fitness'],
        'mean_fitness': history['mean_fitness'],
        'worst_fitness': history['worst_fitness'],
    }).to_csv(os.path.join(rdir, 'fitness_history.csv'), index=False)

    pd.DataFrame({
        'generation': history['generation'],
        'mean_run': history['mean_run'],   'std_run': history['std_run'],
        'mean_short': history['mean_short'], 'std_short': history['std_short'],
        'mean_deep': history['mean_deep'],  'std_deep': history['std_deep'],
        'mean_4th': history['mean_4th'],   'std_4th': history['std_4th'],
    }).to_csv(os.path.join(rdir, 'parameter_evolution.csv'), index=False)

    arr = np.array(population)
    pd.DataFrame({
        'run_prob': arr[:, 0], 'short_pass_prob': arr[:, 1],
        'deep_pass_prob': arr[:, 2], 'fourth_down_aggression': arr[:, 3],
        'fitness': fitnesses,
    }).sort_values('fitness', ascending=False).reset_index(drop=True) \
      .to_csv(os.path.join(rdir, 'final_population.csv'), index=False)

    print(f"  Saved CSVs → results/{run_label}/")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _make_fitness_fig(run_label, gens, best, mean, worst):
    fdir = _figures_dir(run_label)
    os.makedirs(fdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best,  color='#2ecc71', lw=2.5, label='Best')
    ax.plot(gens, mean,  color='#3498db', lw=2.5, label='Mean')
    ax.plot(gens, worst, color='#e74c3c', lw=2.0, linestyle='--', label='Worst')
    ax.fill_between(gens, worst, best, alpha=0.10, color='#3498db')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Fitness', fontsize=13)
    ax.set_title(f'Fitness Over Generations — {run_label.capitalize()}', fontsize=15)
    ax.legend(fontsize=12); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fdir, 'fitness_over_generations.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")


def _make_strategy_fig(run_label, gens, mean_run, mean_short, mean_deep, mean_4th):
    fdir = _figures_dir(run_label)
    os.makedirs(fdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, mean_run,   color='#e67e22', lw=2.5, label='run_prob')
    ax.plot(gens, mean_short, color='#3498db', lw=2.5, label='short_pass_prob')
    ax.plot(gens, mean_deep,  color='#9b59b6', lw=2.5, label='deep_pass_prob')
    ax.plot(gens, mean_4th,   color='#2ecc71', lw=2.0, linestyle='--',
            label='fourth_down_aggression')
    ax.axhline(1/3, color='gray', lw=1.0, linestyle=':', alpha=0.5,
               label='Equal play probs (0.333)')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Mean parameter value', fontsize=13)
    ax.set_title(f'Strategy Parameter Evolution — {run_label.capitalize()}', fontsize=15)
    ax.legend(fontsize=11); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fdir, 'strategy_parameter_evolution.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")


def make_figures(run_label, history):
    gens = history['generation']
    _make_fitness_fig(run_label, gens,
                      history['best_fitness'], history['mean_fitness'],
                      history['worst_fitness'])
    _make_strategy_fig(run_label, gens,
                       history['mean_run'], history['mean_short'],
                       history['mean_deep'], history['mean_4th'])


# ---------------------------------------------------------------------------
# Backfill missing strategy_parameter_evolution.png for run1/run2/run3
# ---------------------------------------------------------------------------
def backfill_missing_figures():
    print("\n--- Auditing figures for run1/run2/run3 ---")
    for run in ('run1', 'run2', 'run3'):
        req = os.path.join(_figures_dir(run), 'strategy_parameter_evolution.png')
        if os.path.exists(req):
            print(f"  {run}: strategy_parameter_evolution.png already exists — skipping")
            continue
        csv = os.path.join(_results_dir(run), 'parameter_evolution.csv')
        if not os.path.exists(csv):
            print(f"  {run}: parameter_evolution.csv missing — cannot generate figure")
            continue
        df = pd.read_csv(csv)
        _make_strategy_fig(
            run, df['generation'].tolist(),
            df['mean_run'].tolist(), df['mean_short'].tolist(),
            df['mean_deep'].tolist(), df['mean_4th'].tolist(),
        )


# ---------------------------------------------------------------------------
# Seed metadata for run1/run2/run3 (known from prior runs)
# ---------------------------------------------------------------------------
KNOWN_SEEDS = {
    'run1': 1695073120,
    'run2': 42,
    'run3': 1893890144,
}

def save_known_seeds():
    for run, seed in KNOWN_SEEDS.items():
        path = os.path.join(_results_dir(run), 'seed.txt')
        if not os.path.exists(path):
            os.makedirs(_results_dir(run), exist_ok=True)
            with open(path, 'w') as f:
                f.write(str(seed) + '\n')


def _load_seed(run):
    path = os.path.join(_results_dir(run), 'seed.txt')
    if os.path.exists(path):
        return int(open(path).read().strip())
    return KNOWN_SEEDS.get(run, 'unknown')


# ---------------------------------------------------------------------------
# Full comparison table
# ---------------------------------------------------------------------------
RUN_META = {
    'run1':  ('mean(pts)',                        'No',  'No',  'No' ),
    'run2':  ('mean(pts) − 0.5·std(pts)',         'No',  'Yes', 'No' ),
    'run3':  ('mean(pts) − 0.5·std(pts)',         'No',  'Yes', 'Yes'),
    'run4a': ('mean(pts)  [TO=−3.5]',             'Yes', 'No',  'No' ),
    'run4b': ('mean(pts) − 0.5·std  [TO=−3.5]',  'Yes', 'Yes', 'No' ),
}

def print_comparison():
    print("\n" + "=" * 110)
    print("  FULL CROSS-RUN COMPARISON")
    print("=" * 110)
    hdr = (f"  {'Run':<6}  {'Fitness Function':<36}  {'TO':>4}  {'Var':>4}  {'Con':>4}  "
           f"{'Best':>7}  {'Mean':>7}  "
           f"{'run%':>5}  {'sht%':>5}  {'dep%':>5}  {'4th':>5}  Seed")
    print(hdr)
    print("  " + "-" * 106)

    for run in ('run1', 'run2', 'run3', 'run4a', 'run4b'):
        fcsv = os.path.join(_results_dir(run), 'fitness_history.csv')
        pcsv = os.path.join(_results_dir(run), 'final_population.csv')
        if not os.path.exists(fcsv):
            print(f"  {run:<6}  (no data)")
            continue

        fdf  = pd.read_csv(fcsv)
        last = fdf.iloc[-1]
        seed = _load_seed(run)

        fn, to, var, con = RUN_META.get(run, ('?', '?', '?', '?'))

        if os.path.exists(pcsv):
            top = pd.read_csv(pcsv).iloc[0]
            r_pct  = top['run_prob'] * 100
            s_pct  = top['short_pass_prob'] * 100
            d_pct  = top['deep_pass_prob'] * 100
            fourth = top['fourth_down_aggression']
        else:
            r_pct = s_pct = d_pct = fourth = float('nan')

        print(f"  {run:<6}  {fn:<36}  {to:>4}  {var:>4}  {con:>4}  "
              f"{last['best_fitness']:>7.3f}  {last['mean_fitness']:>7.3f}  "
              f"{r_pct:>5.1f}  {s_pct:>5.1f}  {d_pct:>5.1f}  {fourth:>5.3f}  {seed}")

    print("=" * 110)
    print("  TO=Turnover penalty  Var=Variance penalty  Con=Situational constraints")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Loading probability distributions (shared across 4a and 4b)...")
    probs = load_probabilities()
    print("  Done.\n")

    save_known_seeds()

    # ---- Run 4a ----
    print("=" * 75)
    print("  RUN 4a — Raw fitness + Turnover penalty (−3.5 per TO), no constraints")
    print("=" * 75, "\n")
    h4a, bg4a, pop4a, fits4a, seed4a = run_ga(
        'run4a',
        variance_penalty=False,
        constraints=False,
        turnover_penalty=TO_PENALTY,
        probs=probs,
    )
    print("\n--- Saving run4a ---")
    save_results('run4a', h4a, bg4a, pop4a, fits4a)
    make_figures('run4a', h4a)

    # ---- Run 4b ----
    print("\n" + "=" * 75)
    print("  RUN 4b — Variance penalty + Turnover penalty (−3.5 per TO), no constraints")
    print("=" * 75, "\n")
    h4b, bg4b, pop4b, fits4b, seed4b = run_ga(
        'run4b',
        variance_penalty=True,
        constraints=False,
        turnover_penalty=TO_PENALTY,
        probs=probs,
    )
    print("\n--- Saving run4b ---")
    save_results('run4b', h4b, bg4b, pop4b, fits4b)
    make_figures('run4b', h4b)

    # ---- Backfill missing figures for run1/2/3 ----
    backfill_missing_figures()

    # ---- Comparison ----
    print_comparison()
    print("\nDone.")

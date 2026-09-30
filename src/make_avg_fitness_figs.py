"""
make_avg_fitness_figs.py — Re-run Run 1, 3, 4a, 4b each for N_SEEDS seeds,
record best-fitness per generation, then plot mean ± 1 std across seeds.

Outputs
-------
results/multi_seed/run{X}/gen_histories.csv
    Columns: generation, seed_1_best, seed_2_best, ..., seed_N_best
    (raw per-seed generation histories — same seeds as gen_seed_strategies.csv)

results/multi_seed/run{X}/gen_stats.csv
    Columns: generation, mean_best, std_best
    (aggregated, ready for plotting)

results/multi_seed/run{X}/gen_seed_strategies.csv
    Columns: seed_idx, seed, best_fitness, converged_run_prob,
             converged_short_pass_prob, converged_deep_pass_prob,
             converged_fourth_down_aggression
    (final converged strategy for each of the N_SEEDS runs — same run set as
     gen_histories.csv so all three files share the same seed ordering)

figures/multi_seed/run{X}_avg_fitness.png
    Mean best fitness ± 1 std across seeds, per generation.
"""

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
POP_SIZE     = 100
N_GEN        = 100
N_DRIVES     = 100
ELITE_N      = 5
TOURNAMENT_K = 5
MUT_STD      = 0.05
N_SEEDS      = 10

_BASE       = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_MS_RESULTS = os.path.join(_BASE, 'results', 'multi_seed')
_MS_FIGURES = os.path.join(_BASE, 'figures', 'multi_seed')

EXPERIMENTS = [
    {
        'label':            'run1',
        'title':            'Run 1 — mean(pts), no constraints',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': 0.0,
        'color':            '#2ecc71',
    },
    {
        'label':            'run3',
        'title':            'Run 3 — mean(pts) − 0.5·std + constraints',
        'variance_penalty': True,
        'constraints':      True,
        'turnover_penalty': 0.0,
        'color':            '#9b59b6',
    },
    {
        'label':            'run4a',
        'title':            'Run 4a — mean(pts), turnover penalty −3.5',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'color':            '#e67e22',
    },
    {
        'label':            'run4b',
        'title':            'Run 4b — mean(pts) − 0.5·std, turnover penalty −3.5',
        'variance_penalty': True,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'color':            '#e74c3c',
    },
]


# ---------------------------------------------------------------------------
# Single GA run — returns generation history AND final converged strategy
# ---------------------------------------------------------------------------
def run_single_with_history(variance_penalty: bool, constraints: bool,
                             turnover_penalty: float, probs: dict) -> tuple:
    """
    Run one GA. Returns:
        seed          (int)
        best_per_gen  (np.ndarray, shape N_GEN+1)
        strategy_row  (dict) — converged strategy of the best individual at gen 100
    """
    seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)

    population   = random_population(POP_SIZE)
    best_per_gen = np.empty(N_GEN + 1)

    for gen in range(N_GEN):
        fitnesses = evaluate_population(
            population, n_drives=N_DRIVES, probs=probs,
            constraints=constraints,
            variance_penalty=variance_penalty,
            turnover_penalty=turnover_penalty,
        )
        best_per_gen[gen] = max(fitnesses)
        population = evolve_generation(
            population, fitnesses,
            elite_n=ELITE_N, tournament_k=TOURNAMENT_K, mutation_std=MUT_STD,
        )

    # Final evaluation at generation N_GEN
    fitnesses = evaluate_population(
        population, n_drives=N_DRIVES, probs=probs,
        constraints=constraints,
        variance_penalty=variance_penalty,
        turnover_penalty=turnover_penalty,
    )
    best_per_gen[N_GEN] = max(fitnesses)

    best_idx   = int(np.argmax(fitnesses))
    best_strat = genome_to_strategy(population[best_idx])

    strategy_row = {
        'seed':                             seed,
        'best_fitness':                     float(best_per_gen[N_GEN]),
        'converged_run_prob':               best_strat.run_prob,
        'converged_short_pass_prob':        best_strat.short_pass_prob,
        'converged_deep_pass_prob':         best_strat.deep_pass_prob,
        'converged_fourth_down_aggression': best_strat.fourth_down_aggression,
    }

    return seed, best_per_gen, strategy_row


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_avg_fitness_figure(label: str, title: str, color: str,
                             gens: np.ndarray, mean_best: np.ndarray,
                             std_best: np.ndarray):
    os.makedirs(_MS_FIGURES, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gens, mean_best, color=color, lw=2.5,
            label=f'Mean best fitness (n={N_SEEDS} seeds)')
    ax.fill_between(gens,
                    mean_best - std_best,
                    mean_best + std_best,
                    alpha=0.25, color=color, label='± 1 std across seeds')

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Best Fitness (pts / drive)', fontsize=13)
    ax.set_title(f'Avg Fitness Over Generations — {title}', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(_MS_FIGURES, f'{label}_avg_fitness.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wall_start = time.time()

    print("=" * 72)
    print(f"  Avg Fitness Figures — {len(EXPERIMENTS)} runs × {N_SEEDS} seeds")
    print(f"  Pop={POP_SIZE}  Gens={N_GEN}  Drives/eval={N_DRIVES}")
    print("=" * 72)

    print("\nLoading probability distributions...")
    probs = load_probabilities()
    print("  Done.\n")

    gens = np.arange(N_GEN + 1)

    for exp in EXPERIMENTS:
        label        = exp['label']
        title        = exp['title']
        color        = exp['color']
        var_pen      = exp['variance_penalty']
        con          = exp['constraints']
        to_pen       = exp['turnover_penalty']

        rdir = os.path.join(_MS_RESULTS, label)
        os.makedirs(rdir, exist_ok=True)

        print(f"{'='*72}")
        print(f"  {label.upper()} — {title}")
        print(f"{'='*72}")
        print(f"  {'Seed':>4}  {'FinalBest':>10}  {'run%':>8}  {'sht%':>8}  {'dep%':>8}  {'4th':>7}  elapsed")
        print(f"  {'-'*70}")

        exp_start      = time.time()
        all_histories  = np.empty((N_SEEDS, N_GEN + 1))
        strategy_rows  = []

        for s_idx in range(N_SEEDS):
            t0                        = time.time()
            seed, history, strat_row  = run_single_with_history(
                variance_penalty=var_pen,
                constraints=con,
                turnover_penalty=to_pen,
                probs=probs,
            )
            all_histories[s_idx] = history
            strat_row['seed_idx'] = s_idx + 1
            strategy_rows.append(strat_row)
            elapsed = time.time() - t0
            m, sec  = divmod(int(elapsed), 60)
            print(
                f"  {s_idx+1:>4}  {history[-1]:>10.4f}  "
                f"run={strat_row['converged_run_prob']*100:>5.1f}%  "
                f"sht={strat_row['converged_short_pass_prob']*100:>5.1f}%  "
                f"dep={strat_row['converged_deep_pass_prob']*100:>5.1f}%  "
                f"4th={strat_row['converged_fourth_down_aggression']:.3f}  "
                f"{m:02d}:{sec:02d}"
            )
            sys.stdout.flush()

        # --- save raw per-seed generation histories ---
        hist_df = pd.DataFrame(
            all_histories.T,
            columns=[f'seed_{i+1}_best' for i in range(N_SEEDS)],
        )
        hist_df.insert(0, 'generation', gens)
        hist_csv = os.path.join(rdir, 'gen_histories.csv')
        hist_df.to_csv(hist_csv, index=False)
        print(f"\n  Saved {hist_csv}")

        # --- save per-seed converged strategies ---
        strat_df = pd.DataFrame(strategy_rows)[
            ['seed_idx', 'seed', 'best_fitness',
             'converged_run_prob', 'converged_short_pass_prob',
             'converged_deep_pass_prob', 'converged_fourth_down_aggression']
        ]
        strat_csv = os.path.join(rdir, 'gen_seed_strategies.csv')
        strat_df.to_csv(strat_csv, index=False)
        print(f"  Saved {strat_csv}")

        # --- save aggregated stats ---
        mean_best = all_histories.mean(axis=0)
        std_best  = all_histories.std(axis=0)
        stats_df  = pd.DataFrame({
            'generation': gens,
            'mean_best':  mean_best,
            'std_best':   std_best,
        })
        stats_csv = os.path.join(rdir, 'gen_stats.csv')
        stats_df.to_csv(stats_csv, index=False)
        print(f"  Saved {stats_csv}")

        # --- figure ---
        make_avg_fitness_figure(label, title, color, gens, mean_best, std_best)

        print(f"  {label} complete — {(time.time()-exp_start)/60:.1f} min\n")

    print(f"All done — total wall time: {(time.time()-wall_start)/60:.1f} min")

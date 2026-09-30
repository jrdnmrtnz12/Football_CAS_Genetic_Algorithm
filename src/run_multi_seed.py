"""
run_multi_seed.py — Run all five experiments each across 10 independent random seeds.

Experiment configurations
-------------------------
run1  : f = mean(pts)                        constraints=False  TO=0.0
run2  : f = mean(pts) − 0.5·std(pts)        constraints=False  TO=0.0
run3  : f = mean(pts) − 0.5·std(pts)        constraints=True   TO=0.0
run4a : f = mean(pts)               [TO=-3.5] constraints=False
run4b : f = mean(pts) − 0.5·std    [TO=-3.5] constraints=False

Outputs
-------
results/multi_seed/run{X}/
    seed_results.csv   — one row per seed: seed, best/mean/worst fitness, converged params
    summary_stats.csv  — one row: mean±std of fitness and params across 10 seeds

results/multi_seed/
    all_runs_summary.csv  — five rows (one per run), same statistics

figures/multi_seed/
    mean_fitness_comparison.png  — bar chart: mean best fitness ± 1 std, all five runs
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
# GA hyper-parameters  (identical to single-run experiments)
# ---------------------------------------------------------------------------
POP_SIZE     = 100
N_GEN        = 100
N_DRIVES     = 100
ELITE_N      = 5
TOURNAMENT_K = 5
MUT_STD      = 0.05
N_SEEDS      = 10

_BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_MS_RESULTS  = os.path.join(_BASE, 'results', 'multi_seed')
_MS_FIGURES  = os.path.join(_BASE, 'figures', 'multi_seed')

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        'label':            'run1',
        'display':          'Run 1',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': 0.0,
        'fitness_fn':       'mean(pts)',
    },
    {
        'label':            'run2',
        'display':          'Run 2',
        'variance_penalty': True,
        'constraints':      False,
        'turnover_penalty': 0.0,
        'fitness_fn':       'mean(pts) − 0.5·std',
    },
    {
        'label':            'run3',
        'display':          'Run 3',
        'variance_penalty': True,
        'constraints':      True,
        'turnover_penalty': 0.0,
        'fitness_fn':       'mean(pts) − 0.5·std + constraints',
    },
    {
        'label':            'run4a',
        'display':          'Run 4a',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'fitness_fn':       'mean(pts)  [TO=−3.5]',
    },
    {
        'label':            'run4b',
        'display':          'Run 4b',
        'variance_penalty': True,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'fitness_fn':       'mean(pts) − 0.5·std  [TO=−3.5]',
    },
]


# ---------------------------------------------------------------------------
# Single GA run — returns summary row dict
# ---------------------------------------------------------------------------
def run_single(variance_penalty: bool, constraints: bool,
               turnover_penalty: float, probs: dict) -> dict:
    """Run one GA with a fresh random seed. Returns a result dict."""
    seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)

    population = random_population(POP_SIZE)

    for _ in range(N_GEN):
        fitnesses = evaluate_population(
            population, n_drives=N_DRIVES, probs=probs,
            constraints=constraints,
            variance_penalty=variance_penalty,
            turnover_penalty=turnover_penalty,
        )
        population = evolve_generation(
            population, fitnesses,
            elite_n=ELITE_N, tournament_k=TOURNAMENT_K, mutation_std=MUT_STD,
        )

    # Final evaluation
    fitnesses = evaluate_population(
        population, n_drives=N_DRIVES, probs=probs,
        constraints=constraints,
        variance_penalty=variance_penalty,
        turnover_penalty=turnover_penalty,
    )

    best_idx    = int(np.argmax(fitnesses))
    best_genome = population[best_idx]
    best_strat  = genome_to_strategy(best_genome)

    return {
        'seed':                       seed,
        'best_fitness':               float(max(fitnesses)),
        'mean_fitness':               float(np.mean(fitnesses)),
        'worst_fitness':              float(min(fitnesses)),
        'converged_run_prob':         best_strat.run_prob,
        'converged_short_pass_prob':  best_strat.short_pass_prob,
        'converged_deep_pass_prob':   best_strat.deep_pass_prob,
        'converged_fourth_down_aggression': best_strat.fourth_down_aggression,
    }


# ---------------------------------------------------------------------------
# Summary statistics from per-seed rows
# ---------------------------------------------------------------------------
def _summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    summary = {'n_seeds': len(df)}
    for col in ['best_fitness', 'mean_fitness', 'worst_fitness',
                'converged_run_prob', 'converged_short_pass_prob',
                'converged_deep_pass_prob', 'converged_fourth_down_aggression']:
        summary[f'mean_{col}'] = float(df[col].mean())
        summary[f'std_{col}']  = float(df[col].std())
    return summary


# ---------------------------------------------------------------------------
# Bar-chart figure
# ---------------------------------------------------------------------------
def make_comparison_figure(run_summaries: list[dict]):
    """Bar chart: mean best fitness ± 1 std for each experiment."""
    os.makedirs(_MS_FIGURES, exist_ok=True)

    labels  = [r['display']            for r in run_summaries]
    means   = [r['mean_best_fitness']  for r in run_summaries]
    stds    = [r['std_best_fitness']   for r in run_summaries]
    colors  = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='white', linewidth=0.8, alpha=0.85,
                  error_kw={'elinewidth': 2.0, 'ecolor': '#333333', 'capthick': 2.0})

    # Annotate mean value above each bar
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.03,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Mean Best Fitness (pts / drive)', fontsize=13)
    ax.set_title(
        f'Mean Best Fitness ± 1 Std Across {N_SEEDS} Seeds — All Experiments',
        fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=min(0, min(m - s for m, s in zip(means, stds))) - 0.1)

    # Legend mapping color → experiment
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, fontsize=11)

    plt.tight_layout()
    path = os.path.join(_MS_FIGURES, 'mean_fitness_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Print final summary table
# ---------------------------------------------------------------------------
def print_summary_table(run_summaries: list[dict]):
    print("\n" + "=" * 112)
    print(f"  MULTI-SEED SUMMARY ({N_SEEDS} seeds per experiment)")
    print("=" * 112)
    hdr = (f"  {'Run':<7}  {'Fitness Function':<38}  "
           f"{'BestFit μ':>9}  {'±σ':>6}  "
           f"{'run% μ':>7}  {'sht% μ':>7}  {'dep% μ':>7}  {'4th μ':>6}")
    print(hdr)
    print("  " + "-" * 108)
    for r in run_summaries:
        print(
            f"  {r['label']:<7}  {r['fitness_fn']:<38}  "
            f"{r['mean_best_fitness']:>9.4f}  {r['std_best_fitness']:>6.4f}  "
            f"{r['mean_converged_run_prob']*100:>7.1f}  "
            f"{r['mean_converged_short_pass_prob']*100:>7.1f}  "
            f"{r['mean_converged_deep_pass_prob']*100:>7.1f}  "
            f"{r['mean_converged_fourth_down_aggression']:>6.3f}"
        )
    print("=" * 112)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wall_start = time.time()

    print("=" * 80)
    print(f"  CAS Football GA — Multi-Seed Experiment")
    print(f"  {len(EXPERIMENTS)} runs × {N_SEEDS} seeds = {len(EXPERIMENTS) * N_SEEDS} total GA runs")
    print(f"  Each GA: Pop={POP_SIZE}  Gens={N_GEN}  Drives/eval={N_DRIVES}")
    print("=" * 80)

    print("\nLoading probability distributions from disk...")
    probs = load_probabilities()
    print("  Done.\n")

    all_runs_rows  = []   # one row per experiment for all_runs_summary.csv
    run_summaries  = []   # for the bar chart and print_summary_table

    for exp in EXPERIMENTS:
        label        = exp['label']
        display      = exp['display']
        var_pen      = exp['variance_penalty']
        con          = exp['constraints']
        to_pen       = exp['turnover_penalty']
        fitness_fn   = exp['fitness_fn']

        rdir = os.path.join(_MS_RESULTS, label)
        os.makedirs(rdir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"  {display} | {fitness_fn}")
        print(f"{'='*80}")
        print(f"  {'Seed':>3}  {'Best':>8}  {'Mean':>8}  "
              f"{'run%':>6}  {'sht%':>6}  {'dep%':>6}  {'4th':>6}  elapsed")
        print(f"  {'-'*68}")

        exp_start = time.time()
        seed_rows = []

        for s_idx in range(1, N_SEEDS + 1):
            t0  = time.time()
            row = run_single(
                variance_penalty=var_pen,
                constraints=con,
                turnover_penalty=to_pen,
                probs=probs,
            )
            elapsed = time.time() - t0
            m, sec  = divmod(int(elapsed), 60)

            seed_rows.append(row)
            print(
                f"  {s_idx:>3}  "
                f"{row['best_fitness']:>8.4f}  {row['mean_fitness']:>8.4f}  "
                f"{row['converged_run_prob']*100:>6.1f}  "
                f"{row['converged_short_pass_prob']*100:>6.1f}  "
                f"{row['converged_deep_pass_prob']*100:>6.1f}  "
                f"{row['converged_fourth_down_aggression']:>6.3f}  "
                f"{m:02d}:{sec:02d}"
            )
            sys.stdout.flush()

        # --- save per-seed CSV ---
        seed_df = pd.DataFrame(seed_rows)
        seed_csv = os.path.join(rdir, 'seed_results.csv')
        seed_df.to_csv(seed_csv, index=False)
        print(f"\n  Saved {seed_csv}")

        # --- summary stats ---
        stats = _summarise(seed_rows)
        summary_row = {
            'run':                                label,
            'display':                            display,
            'fitness_fn':                         fitness_fn,
            'n_seeds':                            N_SEEDS,
            'mean_best_fitness':                  stats['mean_best_fitness'],
            'std_best_fitness':                   stats['std_best_fitness'],
            'mean_mean_fitness':                  stats['mean_mean_fitness'],
            'std_mean_fitness':                   stats['std_mean_fitness'],
            'mean_converged_run_prob':            stats['mean_converged_run_prob'],
            'std_converged_run_prob':             stats['std_converged_run_prob'],
            'mean_converged_short_pass_prob':     stats['mean_converged_short_pass_prob'],
            'std_converged_short_pass_prob':      stats['std_converged_short_pass_prob'],
            'mean_converged_deep_pass_prob':      stats['mean_converged_deep_pass_prob'],
            'std_converged_deep_pass_prob':       stats['std_converged_deep_pass_prob'],
            'mean_converged_fourth_down_aggression': stats['mean_converged_fourth_down_aggression'],
            'std_converged_fourth_down_aggression':  stats['std_converged_fourth_down_aggression'],
        }
        pd.DataFrame([summary_row]).to_csv(
            os.path.join(rdir, 'summary_stats.csv'), index=False)
        print(f"  Saved {os.path.join(rdir, 'summary_stats.csv')}")

        exp_mins = (time.time() - exp_start) / 60
        print(f"  {display} complete — {exp_mins:.1f} min total")

        all_runs_rows.append(summary_row)
        run_summaries.append({**exp, **summary_row})

    # --- aggregate CSV across all 5 runs ---
    all_csv = os.path.join(_MS_RESULTS, 'all_runs_summary.csv')
    pd.DataFrame(all_runs_rows).to_csv(all_csv, index=False)
    print(f"\nSaved {all_csv}")

    # --- figure ---
    print("\n--- Generating comparison figure ---")
    make_comparison_figure(run_summaries)

    # --- table ---
    print_summary_table(run_summaries)

    total_mins = (time.time() - wall_start) / 60
    print(f"\nAll done — total wall time: {total_mins:.1f} min")

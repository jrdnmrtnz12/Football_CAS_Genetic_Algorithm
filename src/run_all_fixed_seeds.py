"""
run_all_fixed_seeds.py — Run all 6 experiments (run1–run5) across the same
10 fixed random seeds so results are directly comparable.

A master seed generates the 10 fixed seeds once. Every experiment uses
np.random.seed(fixed_seed) before random_population(), giving each
experiment the same initial population per seed trial.

Outputs (all in results/multi_seed/ and figures/multi_seed/)
-------------------------------------------------------------
results/multi_seed/fixed_seeds.txt
    The 10 fixed seeds used (for reproducibility)

results/multi_seed/{run}/
    seed_results.csv        — 10 rows: seed, fitness, converged strategy
    gen_histories.csv       — generation × seed matrix of best fitness
    gen_stats.csv           — mean_best + std_best per generation
    gen_seed_strategies.csv — converged strategy per seed
    summary_stats.csv       — aggregate mean/std across 10 seeds

results/multi_seed/
    all_runs_summary.csv    — one row per experiment
    converged_strategy_summary.csv — averaged play-calling strategy per run

figures/multi_seed/
    {run}_avg_fitness.png   — mean ± 1 std over generations
    mean_fitness_comparison.png — bar chart across all 6 runs
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
from update_figures import (
    make_avg_fitness_fig, make_comparison_bar,
    T_SZ, AL_SZ, TK_SZ, LG_SZ, MS_COLORS,
)

# ---------------------------------------------------------------------------
# Fixed seeds — generated once from a master seed
# ---------------------------------------------------------------------------
MASTER_SEED  = 2026
_rng         = np.random.RandomState(MASTER_SEED)
FIXED_SEEDS  = [int(_rng.randint(0, 2**31 - 1)) for _ in range(10)]

# ---------------------------------------------------------------------------
# GA hyper-parameters
# ---------------------------------------------------------------------------
POP_SIZE     = 100
N_GEN        = 100
N_DRIVES     = 100
ELITE_N      = 5
TOURNAMENT_K = 5
MUT_STD      = 0.05
N_SEEDS      = 10

_BASE    = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_MS_RES  = os.path.join(_BASE, 'results', 'multi_seed')
_MS_FIG  = os.path.join(_BASE, 'figures', 'multi_seed')

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        'label':            'run1',
        'display':          'Run 1',
        'fitness_fn':       'mean(pts)',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': 0.0,
        'field_position_epa': False,
    },
    {
        'label':            'run2',
        'display':          'Run 2',
        'fitness_fn':       'mean(pts) − 0.5·std',
        'variance_penalty': True,
        'constraints':      False,
        'turnover_penalty': 0.0,
        'field_position_epa': False,
    },
    {
        'label':            'run3',
        'display':          'Run 3',
        'fitness_fn':       'mean(pts) − 0.5·std + constraints',
        'variance_penalty': True,
        'constraints':      True,
        'turnover_penalty': 0.0,
        'field_position_epa': False,
    },
    {
        'label':            'run4a',
        'display':          'Run 4a',
        'fitness_fn':       'mean(pts)  [TO=−3.5]',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'field_position_epa': False,
    },
    {
        'label':            'run4b',
        'display':          'Run 4b',
        'fitness_fn':       'mean(pts) − 0.5·std  [TO=−3.5]',
        'variance_penalty': True,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'field_position_epa': False,
    },
    {
        'label':            'run5',
        'display':          'Run 5',
        'fitness_fn':       'mean(pts)  [TO=−3.5 + FP-EPA]',
        'variance_penalty': False,
        'constraints':      False,
        'turnover_penalty': -3.5,
        'field_position_epa': True,
    },
]


# ---------------------------------------------------------------------------
# Single GA run with a fixed seed — returns history + converged strategy
# ---------------------------------------------------------------------------
def run_single_fixed(seed: int, variance_penalty: bool, constraints: bool,
                     turnover_penalty: float, field_position_epa: bool,
                     probs: dict) -> tuple:
    """
    Run one GA with a specific seed.
    Returns (seed, best_per_gen np.ndarray[N_GEN+1], strategy_row dict).
    """
    np.random.seed(seed)
    population   = random_population(POP_SIZE)
    best_per_gen = np.empty(N_GEN + 1)

    for gen in range(N_GEN):
        fitnesses = evaluate_population(
            population, n_drives=N_DRIVES, probs=probs,
            constraints=constraints,
            variance_penalty=variance_penalty,
            turnover_penalty=turnover_penalty,
            field_position_epa=field_position_epa,
        )
        best_per_gen[gen] = max(fitnesses)
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
        field_position_epa=field_position_epa,
    )
    best_per_gen[N_GEN] = max(fitnesses)

    best_idx   = int(np.argmax(fitnesses))
    best_strat = genome_to_strategy(population[best_idx])

    strategy_row = {
        'seed_idx':                         0,   # filled by caller
        'seed':                             seed,
        'best_fitness':                     float(best_per_gen[N_GEN]),
        'mean_fitness':                     float(np.mean(fitnesses)),
        'worst_fitness':                    float(min(fitnesses)),
        'converged_run_prob':               best_strat.run_prob,
        'converged_short_pass_prob':        best_strat.short_pass_prob,
        'converged_deep_pass_prob':         best_strat.deep_pass_prob,
        'converged_fourth_down_aggression': best_strat.fourth_down_aggression,
    }
    return best_per_gen, strategy_row


# ---------------------------------------------------------------------------
# Save per-experiment multi-seed outputs
# ---------------------------------------------------------------------------
def save_experiment(label: str, all_best_per_gen: np.ndarray,
                    strat_rows: list) -> dict:
    rdir = os.path.join(_MS_RES, label)
    os.makedirs(rdir, exist_ok=True)
    gens = np.arange(N_GEN + 1)

    # gen_histories.csv
    hist_df = pd.DataFrame(
        all_best_per_gen.T,
        columns=[f'seed_{i+1}_best' for i in range(N_SEEDS)],
    )
    hist_df.insert(0, 'generation', gens)
    hist_df.to_csv(os.path.join(rdir, 'gen_histories.csv'), index=False)

    # gen_stats.csv
    mean_best = all_best_per_gen.mean(axis=0)
    std_best  = all_best_per_gen.std(axis=0)
    pd.DataFrame({'generation': gens, 'mean_best': mean_best, 'std_best': std_best}
                 ).to_csv(os.path.join(rdir, 'gen_stats.csv'), index=False)

    # seed_results.csv
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(rdir, 'seed_results.csv'), index=False)

    # gen_seed_strategies.csv
    pd.DataFrame(strat_rows)[
        ['seed_idx', 'seed', 'best_fitness',
         'converged_run_prob', 'converged_short_pass_prob',
         'converged_deep_pass_prob', 'converged_fourth_down_aggression']
    ].to_csv(os.path.join(rdir, 'gen_seed_strategies.csv'), index=False)

    # summary_stats.csv
    bf  = [r['best_fitness']                     for r in strat_rows]
    mf  = [r['mean_fitness']                     for r in strat_rows]
    run = [r['converged_run_prob']               for r in strat_rows]
    sht = [r['converged_short_pass_prob']        for r in strat_rows]
    dep = [r['converged_deep_pass_prob']         for r in strat_rows]
    fth = [r['converged_fourth_down_aggression'] for r in strat_rows]

    summary = {
        'run': label, 'n_seeds': N_SEEDS,
        'mean_best_fitness':  np.mean(bf), 'std_best_fitness':  np.std(bf),
        'mean_mean_fitness':  np.mean(mf), 'std_mean_fitness':  np.std(mf),
        'mean_converged_run_prob':               np.mean(run),
        'std_converged_run_prob':                np.std(run),
        'mean_converged_short_pass_prob':        np.mean(sht),
        'std_converged_short_pass_prob':         np.std(sht),
        'mean_converged_deep_pass_prob':         np.mean(dep),
        'std_converged_deep_pass_prob':          np.std(dep),
        'mean_converged_fourth_down_aggression': np.mean(fth),
        'std_converged_fourth_down_aggression':  np.std(fth),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(rdir, 'summary_stats.csv'), index=False)

    print(f'  Saved results/multi_seed/{label}/')
    return summary, mean_best, std_best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wall_start = time.time()

    print('=' * 80)
    print(f'  CAS Football GA — All 6 Experiments × {N_SEEDS} Fixed Seeds')
    print(f'  Master seed: {MASTER_SEED}')
    print(f'  Fixed seeds: {FIXED_SEEDS}')
    print(f'  Each GA: Pop={POP_SIZE}  Gens={N_GEN}  Drives/eval={N_DRIVES}')
    print('=' * 80)

    # Save fixed seeds for reproducibility
    os.makedirs(_MS_RES, exist_ok=True)
    with open(os.path.join(_MS_RES, 'fixed_seeds.txt'), 'w') as f:
        f.write(f'master_seed={MASTER_SEED}\n')
        for i, s in enumerate(FIXED_SEEDS, 1):
            f.write(f'seed_{i}={s}\n')
    print(f'\nSaved results/multi_seed/fixed_seeds.txt')

    print('\nLoading probability distributions...')
    probs = load_probabilities()
    print('  Done.\n')

    all_summaries   = []
    run_summaries   = []   # for bar chart

    for exp in EXPERIMENTS:
        label   = exp['label']
        display = exp['display']
        fn_str  = exp['fitness_fn']

        print(f'{"="*80}')
        print(f'  {display} | {fn_str}')
        print(f'{"="*80}')
        print(f"  {'#':>3}  {'Seed':>12}  {'Best':>8}  "
              f"{'run%':>6}  {'sht%':>6}  {'dep%':>6}  {'4th':>6}  elapsed")
        print(f"  {'-'*68}")

        exp_start        = time.time()
        all_best_per_gen = np.empty((N_SEEDS, N_GEN + 1))
        strat_rows       = []

        for s_idx, seed in enumerate(FIXED_SEEDS):
            t0 = time.time()
            best_per_gen, strat_row = run_single_fixed(
                seed=seed,
                variance_penalty=exp['variance_penalty'],
                constraints=exp['constraints'],
                turnover_penalty=exp['turnover_penalty'],
                field_position_epa=exp['field_position_epa'],
                probs=probs,
            )
            all_best_per_gen[s_idx] = best_per_gen
            strat_row['seed_idx']   = s_idx + 1
            strat_rows.append(strat_row)

            elapsed = time.time() - t0
            m, sec  = divmod(int(elapsed), 60)
            print(
                f"  {s_idx+1:>3}  {seed:>12}  "
                f"{strat_row['best_fitness']:>8.4f}  "
                f"{strat_row['converged_run_prob']*100:>6.1f}  "
                f"{strat_row['converged_short_pass_prob']*100:>6.1f}  "
                f"{strat_row['converged_deep_pass_prob']*100:>6.1f}  "
                f"{strat_row['converged_fourth_down_aggression']:>6.3f}  "
                f"{m:02d}:{sec:02d}"
            )
            sys.stdout.flush()

        summary, mean_best, std_best = save_experiment(
            label, all_best_per_gen, strat_rows)
        summary['display']    = display
        summary['fitness_fn'] = fn_str
        all_summaries.append(summary)

        # avg fitness figure
        gens  = np.arange(N_GEN + 1)
        gs_df = pd.DataFrame({'generation': gens,
                               'mean_best': mean_best, 'std_best': std_best})
        make_avg_fitness_fig(label, gs_df, n_seeds=N_SEEDS)

        exp_mins = (time.time() - exp_start) / 60
        print(f'  {display} done — {exp_mins:.1f} min | '
              f'mean best: {summary["mean_best_fitness"]:.4f} '
              f'± {summary["std_best_fitness"]:.4f}\n')

    # all_runs_summary.csv
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(_MS_RES, 'all_runs_summary.csv'), index=False)
    print(f'Saved results/multi_seed/all_runs_summary.csv')

    # converged_strategy_summary.csv
    strat_summary_rows = []
    for s in all_summaries:
        strat_summary_rows.append({
            'run':                           s['run'],
            'mean_best_fitness':             s['mean_best_fitness'],
            'std_best_fitness':              s['std_best_fitness'],
            'mean_run_prob':                 s['mean_converged_run_prob'],
            'std_run_prob':                  s['std_converged_run_prob'],
            'mean_short_pass_prob':          s['mean_converged_short_pass_prob'],
            'std_short_pass_prob':           s['std_converged_short_pass_prob'],
            'mean_deep_pass_prob':           s['mean_converged_deep_pass_prob'],
            'std_deep_pass_prob':            s['std_converged_deep_pass_prob'],
            'mean_fourth_down_aggression':   s['mean_converged_fourth_down_aggression'],
            'std_fourth_down_aggression':    s['std_converged_fourth_down_aggression'],
        })
    pd.DataFrame(strat_summary_rows).to_csv(
        os.path.join(_MS_RES, 'converged_strategy_summary.csv'), index=False)
    print(f'Saved results/multi_seed/converged_strategy_summary.csv')

    # comparison bar chart
    make_comparison_bar(summary_df)

    # Final summary table
    print('\n' + '=' * 95)
    print(f'  FIXED-SEED SUMMARY (master_seed={MASTER_SEED}, {N_SEEDS} seeds per experiment)')
    print('=' * 95)
    print(f"  {'Run':<6}  {'Fitness Function':<38}  "
          f"{'BestFit μ':>9}  {'±σ':>6}  "
          f"{'run% μ':>7}  {'sht% μ':>7}  {'dep% μ':>7}  {'4th μ':>6}")
    print('  ' + '-' * 91)
    for s in all_summaries:
        print(
            f"  {s['run']:<6}  {s['fitness_fn']:<38}  "
            f"{s['mean_best_fitness']:>9.4f}  {s['std_best_fitness']:>6.4f}  "
            f"{s['mean_converged_run_prob']*100:>7.1f}  "
            f"{s['mean_converged_short_pass_prob']*100:>7.1f}  "
            f"{s['mean_converged_deep_pass_prob']*100:>7.1f}  "
            f"{s['mean_converged_fourth_down_aggression']:>6.3f}"
        )
    print('=' * 95)

    total_mins = (time.time() - wall_start) / 60
    print(f'\nAll done — total wall time: {total_mins:.1f} min')

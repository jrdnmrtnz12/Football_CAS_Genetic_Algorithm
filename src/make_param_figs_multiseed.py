"""
make_param_figs_multiseed.py
Re-runs all 6 experiments × 10 fixed seeds tracking per-generation population
mean parameters. Saves param_stats.csv per run and generates
figures/multi_seed/{run}_strategy_param_evolution.png for all 6 runs.
Also regenerates mean_fitness_comparison.png from existing all_runs_summary.csv.
"""

import os, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from genetic_algorithm import random_population, evaluate_population, evolve_generation
from load_data import load_probabilities
from update_figures import T_SZ, AL_SZ, TK_SZ, LG_SZ, MS_COLORS, MS_TITLES, make_comparison_bar

_BASE   = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_MS_RES = os.path.join(_BASE, 'results', 'multi_seed')
_MS_FIG = os.path.join(_BASE, 'figures', 'multi_seed')

# ---------------------------------------------------------------------------
# Fixed seeds (from file)
# ---------------------------------------------------------------------------
FIXED_SEEDS = [
    942082305, 1145077126, 1773871898, 1980789688, 2047133773,
    1988008269, 381818397, 889207412, 2058534300, 84665779,
]
N_SEEDS = len(FIXED_SEEDS)

# GA hyper-parameters
POP_SIZE     = 100
N_GEN        = 100
N_DRIVES     = 100
ELITE_N      = 5
TOURNAMENT_K = 5
MUT_STD      = 0.05

EXPERIMENTS = [
    {'label': 'run1',  'variance_penalty': False, 'constraints': False, 'turnover_penalty': 0.0,  'field_position_epa': False},
    {'label': 'run2',  'variance_penalty': True,  'constraints': False, 'turnover_penalty': 0.0,  'field_position_epa': False},
    {'label': 'run3',  'variance_penalty': True,  'constraints': True,  'turnover_penalty': 0.0,  'field_position_epa': False},
    {'label': 'run4a', 'variance_penalty': False, 'constraints': False, 'turnover_penalty': -3.5, 'field_position_epa': False},
    {'label': 'run4b', 'variance_penalty': True,  'constraints': False, 'turnover_penalty': -3.5, 'field_position_epa': False},
    {'label': 'run5',  'variance_penalty': False, 'constraints': False, 'turnover_penalty': -3.5, 'field_position_epa': True},
]


def _pop_param_means(population):
    """Return (mean_run, mean_short, mean_deep, mean_4th) for population."""
    arr = np.array(population)   # shape (POP_SIZE, 4)
    return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(), arr[:, 3].mean()


def run_single_fixed_params(seed, variance_penalty, constraints,
                            turnover_penalty, field_position_epa, probs):
    """
    Run one GA with a fixed seed. Returns param_history array of shape
    (N_GEN+1, 4) — [mean_run, mean_short, mean_deep, mean_4th] per generation.
    """
    np.random.seed(seed)
    population   = random_population(POP_SIZE)
    param_hist   = np.empty((N_GEN + 1, 4))
    param_hist[0] = _pop_param_means(population)

    for gen in range(N_GEN):
        fitnesses = evaluate_population(
            population, n_drives=N_DRIVES, probs=probs,
            constraints=constraints,
            variance_penalty=variance_penalty,
            turnover_penalty=turnover_penalty,
            field_position_epa=field_position_epa,
        )
        population = evolve_generation(
            population, fitnesses,
            elite_n=ELITE_N, tournament_k=TOURNAMENT_K, mutation_std=MUT_STD,
        )
        param_hist[gen + 1] = _pop_param_means(population)

    return param_hist


def make_param_fig(run, param_stats_df):
    os.makedirs(_MS_FIG, exist_ok=True)
    gens = param_stats_df['generation']

    fig, ax = plt.subplots(figsize=(10, 6))

    for col, color, label in [
        ('mean_run',   '#e67e22', 'run_prob'),
        ('mean_short', '#3498db', 'short_pass_prob'),
        ('mean_deep',  '#9b59b6', 'deep_pass_prob'),
        ('mean_4th',   '#2ecc71', 'fourth_down_aggression'),
    ]:
        std_col = col.replace('mean_', 'std_')
        ls = '--' if col == 'mean_4th' else '-'
        ax.plot(gens, param_stats_df[col], color=color, lw=2.5,
                linestyle=ls, label=label)
        ax.fill_between(gens,
                        param_stats_df[col] - param_stats_df[std_col],
                        param_stats_df[col] + param_stats_df[std_col],
                        alpha=0.15, color=color)

    ax.axhline(1/3, color='gray', lw=1.0, linestyle=':', alpha=0.5,
               label='Equal play probs (0.333)')
    ax.set_xlabel('Generation', fontsize=AL_SZ)
    ax.set_ylabel('Mean parameter value across population & seeds', fontsize=AL_SZ)
    ax.set_title(f'Strategy Parameter Evolution — {MS_TITLES[run]}', fontsize=T_SZ)
    ax.legend(fontsize=LG_SZ)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=TK_SZ)
    plt.tight_layout()
    path = os.path.join(_MS_FIG, f'{run}_strategy_param_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


if __name__ == '__main__':
    wall_start = time.time()
    print('Loading probability distributions...')
    probs = load_probabilities()
    print('  Done.\n')

    for exp in EXPERIMENTS:
        label = exp['label']
        print(f'{"="*70}')
        print(f'  {label} — tracking parameter evolution across {N_SEEDS} seeds')
        print(f'{"="*70}')
        t0 = time.time()

        # shape: (N_SEEDS, N_GEN+1, 4)
        all_param_hist = np.empty((N_SEEDS, N_GEN + 1, 4))

        for s_idx, seed in enumerate(FIXED_SEEDS):
            all_param_hist[s_idx] = run_single_fixed_params(
                seed=seed,
                variance_penalty=exp['variance_penalty'],
                constraints=exp['constraints'],
                turnover_penalty=exp['turnover_penalty'],
                field_position_epa=exp['field_position_epa'],
                probs=probs,
            )
            sys.stdout.write(f'  seed {s_idx+1}/{N_SEEDS} done\n')
            sys.stdout.flush()

        # Aggregate: mean and std across seeds per generation
        gens        = np.arange(N_GEN + 1)
        mean_params = all_param_hist.mean(axis=0)   # (N_GEN+1, 4)
        std_params  = all_param_hist.std(axis=0)

        param_stats_df = pd.DataFrame({
            'generation': gens,
            'mean_run':   mean_params[:, 0], 'std_run':   std_params[:, 0],
            'mean_short': mean_params[:, 1], 'std_short': std_params[:, 1],
            'mean_deep':  mean_params[:, 2], 'std_deep':  std_params[:, 2],
            'mean_4th':   mean_params[:, 3], 'std_4th':   std_params[:, 3],
        })

        rdir = os.path.join(_MS_RES, label)
        os.makedirs(rdir, exist_ok=True)
        param_stats_df.to_csv(os.path.join(rdir, 'param_stats.csv'), index=False)
        print(f'  Saved results/multi_seed/{label}/param_stats.csv')

        make_param_fig(label, param_stats_df)

        elapsed = (time.time() - t0) / 60
        print(f'  {label} done — {elapsed:.1f} min\n')

    # Regenerate comparison bar chart from existing summary
    print('Regenerating mean_fitness_comparison.png ...')
    sum_csv = os.path.join(_MS_RES, 'all_runs_summary.csv')
    summary_df = pd.read_csv(sum_csv)
    make_comparison_bar(summary_df)

    total_mins = (time.time() - wall_start) / 60
    print(f'\nAll done — total wall time: {total_mins:.1f} min')

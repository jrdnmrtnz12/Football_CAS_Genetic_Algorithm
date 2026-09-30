"""
update_figures.py — Regenerate all existing figures with larger fonts.

Font spec
---------
  Title      : 18
  Axis labels: 16
  Tick labels: 14
  Legend     : 13

Reads from saved CSVs; overwrites existing PNGs.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BASE       = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS    = os.path.join(_BASE, 'results')
_FIGURES    = os.path.join(_BASE, 'figures')
_MS_RES     = os.path.join(_RESULTS, 'multi_seed')
_MS_FIG     = os.path.join(_FIGURES, 'multi_seed')

# ---------------------------------------------------------------------------
# Font constants
# ---------------------------------------------------------------------------
T_SZ  = 18   # title
AL_SZ = 16   # axis label
TK_SZ = 14   # tick label
LG_SZ = 13   # legend


def _ticks(ax):
    ax.tick_params(axis='both', labelsize=TK_SZ)


# ---------------------------------------------------------------------------
# Per-run single-seed figures
# ---------------------------------------------------------------------------

RUN_DISPLAY = {
    'run1':  'Run 1 — Raw Fitness',
    'run2':  'Run 2 — Variance Penalty',
    'run3':  'Run 3 — Variance Penalty + Constraints',
    'run4a': 'Run 4a — Turnover Penalty (−3.5)',
    'run4b': 'Run 4b — Variance + Turnover Penalty',
    'run5':  'Run 5 — Turnover + Field-Position EPA',
}


def make_fitness_fig(run: str, df: pd.DataFrame):
    fdir = os.path.join(_FIGURES, run)
    os.makedirs(fdir, exist_ok=True)
    gens  = df['generation']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, df['best_fitness'],  color='#2ecc71', lw=2.5, label='Best')
    ax.plot(gens, df['mean_fitness'],  color='#3498db', lw=2.5, label='Mean')
    ax.plot(gens, df['worst_fitness'], color='#e74c3c', lw=2.0,
            linestyle='--', label='Worst')
    ax.fill_between(gens, df['worst_fitness'], df['best_fitness'],
                    alpha=0.10, color='#3498db')
    ax.set_xlabel('Generation', fontsize=AL_SZ)
    ax.set_ylabel('Fitness (pts / drive)', fontsize=AL_SZ)
    ax.set_title(f'Fitness Over Generations — {RUN_DISPLAY[run]}', fontsize=T_SZ)
    ax.legend(fontsize=LG_SZ)
    ax.grid(alpha=0.3)
    _ticks(ax)
    plt.tight_layout()
    path = os.path.join(fdir, 'fitness_over_generations.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def make_strategy_fig(run: str, df: pd.DataFrame):
    fdir = os.path.join(_FIGURES, run)
    os.makedirs(fdir, exist_ok=True)
    gens = df['generation']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, df['mean_run'],   color='#e67e22', lw=2.5, label='run_prob')
    ax.plot(gens, df['mean_short'], color='#3498db', lw=2.5, label='short_pass_prob')
    ax.plot(gens, df['mean_deep'],  color='#9b59b6', lw=2.5, label='deep_pass_prob')
    ax.plot(gens, df['mean_4th'],   color='#2ecc71', lw=2.0,
            linestyle='--', label='fourth_down_aggression')
    ax.axhline(1/3, color='gray', lw=1.0, linestyle=':', alpha=0.5,
               label='Equal play probs (0.333)')
    ax.set_xlabel('Generation', fontsize=AL_SZ)
    ax.set_ylabel('Mean parameter value', fontsize=AL_SZ)
    ax.set_title(f'Strategy Parameter Evolution — {RUN_DISPLAY[run]}', fontsize=T_SZ)
    ax.legend(fontsize=LG_SZ)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    _ticks(ax)
    plt.tight_layout()
    path = os.path.join(fdir, 'strategy_parameter_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def make_diversity_fig(run: str, df: pd.DataFrame):
    fdir = os.path.join(_FIGURES, run)
    os.makedirs(fdir, exist_ok=True)
    gens = df['generation']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, df['std_run'],   color='#e67e22', lw=2.5, label='run_prob')
    ax.plot(gens, df['std_short'], color='#3498db', lw=2.5, label='short_pass_prob')
    ax.plot(gens, df['std_deep'],  color='#9b59b6', lw=2.5, label='deep_pass_prob')
    ax.plot(gens, df['std_4th'],   color='#2ecc71', lw=2.0,
            linestyle='--', label='fourth_down_aggression')
    ax.set_xlabel('Generation', fontsize=AL_SZ)
    ax.set_ylabel('Std dev across population', fontsize=AL_SZ)
    ax.set_title(f'Population Diversity Over Generations — {RUN_DISPLAY[run]}', fontsize=T_SZ)
    ax.legend(fontsize=LG_SZ)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    _ticks(ax)
    plt.tight_layout()
    path = os.path.join(fdir, 'population_diversity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Multi-seed: per-run average fitness figure
# ---------------------------------------------------------------------------

MS_COLORS = {
    'run1':  '#2ecc71',
    'run2':  '#3498db',
    'run3':  '#9b59b6',
    'run4a': '#e67e22',
    'run4b': '#e74c3c',
    'run5':  '#1abc9c',
}

MS_TITLES = {
    'run1':  'Run 1 — mean(pts), no constraints',
    'run2':  'Run 2 — mean(pts) − 0.5·std, no constraints',
    'run3':  'Run 3 — mean(pts) − 0.5·std + constraints',
    'run4a': 'Run 4a — mean(pts), turnover penalty −3.5',
    'run4b': 'Run 4b — mean(pts) − 0.5·std, turnover penalty −3.5',
    'run5':  'Run 5 — mean(pts), turnover + field-position EPA',
}


def make_avg_fitness_fig(run: str, df: pd.DataFrame, n_seeds: int = 10):
    os.makedirs(_MS_FIG, exist_ok=True)
    color = MS_COLORS[run]
    gens  = df['generation']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, df['mean_best'], color=color, lw=2.5,
            label=f'Mean best fitness (n={n_seeds} seeds)')
    ax.fill_between(gens,
                    df['mean_best'] - df['std_best'],
                    df['mean_best'] + df['std_best'],
                    alpha=0.25, color=color, label='± 1 std across seeds')
    ax.set_xlabel('Generation', fontsize=AL_SZ)
    ax.set_ylabel('Best Fitness (pts / drive)', fontsize=AL_SZ)
    ax.set_title(f'Avg Fitness Over Generations — {MS_TITLES[run]}', fontsize=T_SZ)
    ax.legend(fontsize=LG_SZ)
    ax.grid(alpha=0.3)
    _ticks(ax)
    plt.tight_layout()
    path = os.path.join(_MS_FIG, f'{run}_avg_fitness.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Multi-seed: comparison bar chart (all runs)
# ---------------------------------------------------------------------------

def make_comparison_bar(summary_df: pd.DataFrame):
    """Bar chart of mean best fitness ± 1 std for each row in summary_df."""
    os.makedirs(_MS_FIG, exist_ok=True)

    labels  = summary_df['display'].tolist()
    means   = summary_df['mean_best_fitness'].tolist()
    stds    = summary_df['std_best_fitness'].tolist()
    runs    = summary_df['run'].tolist()
    colors  = [MS_COLORS.get(r, '#95a5a6') for r in runs]

    fig, ax = plt.subplots(figsize=(11, 6))
    x    = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='white', linewidth=0.8, alpha=0.85,
                  error_kw={'elinewidth': 2.0, 'ecolor': '#333333', 'capthick': 2.0})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.04,
                f'{m:.3f}', ha='center', va='bottom',
                fontsize=TK_SZ, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TK_SZ)
    ax.set_ylabel('Mean Best Fitness (pts / drive)', fontsize=AL_SZ)
    ax.set_title(
        f'Mean Best Fitness ± 1 Std Across {summary_df["n_seeds"].iloc[0]} Seeds — All Experiments',
        fontsize=T_SZ)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=TK_SZ)
    ax.set_ylim(bottom=min(0, min(m - s for m, s in zip(means, stds))) - 0.15)

    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, fontsize=LG_SZ)

    plt.tight_layout()
    path = os.path.join(_MS_FIG, 'mean_fitness_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 60)
    print('  Regenerating all figures with updated fonts')
    print(f'  Title={T_SZ}  AxisLabel={AL_SZ}  Tick={TK_SZ}  Legend={LG_SZ}')
    print('=' * 60)

    # ---- Single-seed run figures (run1 – run4b) ----
    for run in ('run1', 'run2', 'run3', 'run4a', 'run4b'):
        print(f'\n--- {run} ---')
        fit_csv   = os.path.join(_RESULTS, run, 'fitness_history.csv')
        param_csv = os.path.join(_RESULTS, run, 'parameter_evolution.csv')

        if os.path.exists(fit_csv):
            make_fitness_fig(run, pd.read_csv(fit_csv))
        else:
            print(f'  SKIP: {fit_csv} not found')

        if os.path.exists(param_csv):
            df_p = pd.read_csv(param_csv)
            make_strategy_fig(run, df_p)
            if run in ('run1', 'run2', 'run3'):
                make_diversity_fig(run, df_p)
        else:
            print(f'  SKIP: {param_csv} not found')

    # ---- Multi-seed avg fitness figures ----
    print('\n--- multi_seed avg fitness ---')
    for run in ('run1', 'run3', 'run4a', 'run4b'):
        gs_csv = os.path.join(_MS_RES, run, 'gen_stats.csv')
        if os.path.exists(gs_csv):
            make_avg_fitness_fig(run, pd.read_csv(gs_csv))
        else:
            print(f'  SKIP: {gs_csv} not found')

    # ---- Multi-seed comparison bar chart ----
    print('\n--- multi_seed comparison bar chart ---')
    sum_csv = os.path.join(_MS_RES, 'all_runs_summary.csv')
    if os.path.exists(sum_csv):
        make_comparison_bar(pd.read_csv(sum_csv))
    else:
        print(f'  SKIP: {sum_csv} not found')

    print('\nDone.')

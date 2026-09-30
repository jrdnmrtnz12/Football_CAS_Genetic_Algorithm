"""
run_run5.py — Run 5: Raw fitness + turnover penalty (−3.5) + field-position EPA
              for punts and failed 4th-down conversions.

Fitness  : mean(pts)  with TO penalty −3.5 and field-position-adjusted punt/fail EPA
Constraints: False
Variance penalty: False

Runs 10 independent seeds.

Outputs
-------
results/run5/
    fitness_history.csv, parameter_evolution.csv, final_population.csv,
    seed.txt, best_strategy_drives.txt   (best single seed)

results/multi_seed/run5/
    seed_results.csv, summary_stats.csv,
    gen_histories.csv, gen_stats.csv, gen_seed_strategies.csv

figures/run5/
    fitness_over_generations.png, strategy_parameter_evolution.png  (best seed)

figures/multi_seed/
    run5_avg_fitness.png              (averaged across 10 seeds)
    mean_fitness_comparison.png       (all 6 runs, updated)
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
from update_figures import (
    make_fitness_fig, make_strategy_fig, make_avg_fitness_fig, make_comparison_bar,
    T_SZ, AL_SZ, TK_SZ, LG_SZ, MS_COLORS, _ticks,
)

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
TO_PENALTY   = -3.5
FP_EPA       = True     # field-position EPA for punts / failed 4th downs

_BASE    = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
RUN      = 'run5'
R_DIR    = os.path.join(_BASE, 'results', RUN)
F_DIR    = os.path.join(_BASE, 'figures', RUN)
MS_RES   = os.path.join(_BASE, 'results', 'multi_seed', RUN)
MS_FIG   = os.path.join(_BASE, 'figures', 'multi_seed')


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
    return f'{m:02d}:{s:02d}'


# ---------------------------------------------------------------------------
# Single GA run — full history + converged strategy
# ---------------------------------------------------------------------------
def run_single(probs):
    seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    population   = random_population(POP_SIZE)

    gen_hist = {k: [] for k in ['generation', 'best_fitness', 'mean_fitness',
                                 'worst_fitness', 'mean_run', 'std_run',
                                 'mean_short', 'std_short', 'mean_deep',
                                 'std_deep', 'mean_4th', 'std_4th']}
    best_per_gen = np.empty(N_GEN + 1)

    for gen in range(N_GEN):
        fitnesses = evaluate_population(
            population, n_drives=N_DRIVES, probs=probs,
            constraints=False, variance_penalty=False,
            turnover_penalty=TO_PENALTY, field_position_epa=FP_EPA,
        )
        stats = _param_stats(population)
        best_per_gen[gen] = max(fitnesses)

        gen_hist['generation'].append(gen)
        gen_hist['best_fitness'].append(max(fitnesses))
        gen_hist['mean_fitness'].append(float(np.mean(fitnesses)))
        gen_hist['worst_fitness'].append(min(fitnesses))
        for k in ('mean_run', 'std_run', 'mean_short', 'std_short',
                  'mean_deep', 'std_deep', 'mean_4th', 'std_4th'):
            gen_hist[k].append(stats[k])

        population = evolve_generation(
            population, fitnesses,
            elite_n=ELITE_N, tournament_k=TOURNAMENT_K, mutation_std=MUT_STD,
        )

    # Final eval at gen N_GEN
    fitnesses = evaluate_population(
        population, n_drives=N_DRIVES, probs=probs,
        constraints=False, variance_penalty=False,
        turnover_penalty=TO_PENALTY, field_position_epa=FP_EPA,
    )
    stats = _param_stats(population)
    best_per_gen[N_GEN] = max(fitnesses)

    gen_hist['generation'].append(N_GEN)
    gen_hist['best_fitness'].append(max(fitnesses))
    gen_hist['mean_fitness'].append(float(np.mean(fitnesses)))
    gen_hist['worst_fitness'].append(min(fitnesses))
    for k in ('mean_run', 'std_run', 'mean_short', 'std_short',
              'mean_deep', 'std_deep', 'mean_4th', 'std_4th'):
        gen_hist[k].append(stats[k])

    best_idx   = int(np.argmax(fitnesses))
    best_strat = genome_to_strategy(population[best_idx])

    strat_row = {
        'seed':                             seed,
        'best_fitness':                     float(best_per_gen[N_GEN]),
        'mean_fitness':                     float(np.mean(fitnesses)),
        'worst_fitness':                    float(min(fitnesses)),
        'converged_run_prob':               best_strat.run_prob,
        'converged_short_pass_prob':        best_strat.short_pass_prob,
        'converged_deep_pass_prob':         best_strat.deep_pass_prob,
        'converged_fourth_down_aggression': best_strat.fourth_down_aggression,
    }

    return seed, gen_hist, best_per_gen, population[best_idx], strat_row, fitnesses


# ---------------------------------------------------------------------------
# Save canonical single-seed results to results/run5/
# ---------------------------------------------------------------------------
def save_best_seed(gen_hist, best_genome, population, fitnesses, seed):
    os.makedirs(R_DIR, exist_ok=True)

    pd.DataFrame({
        'generation':    gen_hist['generation'],
        'best_fitness':  gen_hist['best_fitness'],
        'mean_fitness':  gen_hist['mean_fitness'],
        'worst_fitness': gen_hist['worst_fitness'],
    }).to_csv(os.path.join(R_DIR, 'fitness_history.csv'), index=False)

    pd.DataFrame({k: gen_hist[k] for k in
                  ['generation', 'mean_run', 'std_run', 'mean_short', 'std_short',
                   'mean_deep', 'std_deep', 'mean_4th', 'std_4th']}
    ).to_csv(os.path.join(R_DIR, 'parameter_evolution.csv'), index=False)

    arr = np.array(population)
    pd.DataFrame({
        'run_prob': arr[:, 0], 'short_pass_prob': arr[:, 1],
        'deep_pass_prob': arr[:, 2], 'fourth_down_aggression': arr[:, 3],
        'fitness': fitnesses,
    }).sort_values('fitness', ascending=False).reset_index(drop=True) \
      .to_csv(os.path.join(R_DIR, 'final_population.csv'), index=False)

    with open(os.path.join(R_DIR, 'seed.txt'), 'w') as f:
        f.write(str(seed) + '\n')

    print(f'  Saved CSVs → results/{RUN}/')


# ---------------------------------------------------------------------------
# Save 10-seed multi_seed results to results/multi_seed/run5/
# ---------------------------------------------------------------------------
def save_multi_seed(all_histories, strat_rows, all_best_per_gen):
    os.makedirs(MS_RES, exist_ok=True)
    gens = np.arange(N_GEN + 1)

    # seed_results.csv  (seed_idx already present in each strat_row dict)
    seed_df = pd.DataFrame(strat_rows)
    seed_df.to_csv(os.path.join(MS_RES, 'seed_results.csv'), index=False)

    # gen_seed_strategies.csv (aligned with gen_histories by seed_idx)
    strat_df = pd.DataFrame(strat_rows)[
        ['seed_idx', 'seed', 'best_fitness', 'converged_run_prob',
         'converged_short_pass_prob', 'converged_deep_pass_prob',
         'converged_fourth_down_aggression']
    ]
    strat_df.to_csv(os.path.join(MS_RES, 'gen_seed_strategies.csv'), index=False)

    # gen_histories.csv
    hist_df = pd.DataFrame(
        all_best_per_gen.T,
        columns=[f'seed_{i+1}_best' for i in range(N_SEEDS)],
    )
    hist_df.insert(0, 'generation', gens)
    hist_df.to_csv(os.path.join(MS_RES, 'gen_histories.csv'), index=False)

    # gen_stats.csv
    mean_best = all_best_per_gen.mean(axis=0)
    std_best  = all_best_per_gen.std(axis=0)
    pd.DataFrame({
        'generation': gens, 'mean_best': mean_best, 'std_best': std_best,
    }).to_csv(os.path.join(MS_RES, 'gen_stats.csv'), index=False)

    # summary_stats.csv
    bf   = [r['best_fitness'] for r in strat_rows]
    mf   = [r['mean_fitness']  for r in strat_rows]
    run  = [r['converged_run_prob']               for r in strat_rows]
    sht  = [r['converged_short_pass_prob']        for r in strat_rows]
    dep  = [r['converged_deep_pass_prob']         for r in strat_rows]
    fth  = [r['converged_fourth_down_aggression'] for r in strat_rows]

    pd.DataFrame([{
        'run': RUN, 'display': 'Run 5', 'n_seeds': N_SEEDS,
        'fitness_fn': 'mean(pts) [TO=−3.5 + field-pos EPA]',
        'mean_best_fitness':  np.mean(bf),  'std_best_fitness':  np.std(bf),
        'mean_mean_fitness':  np.mean(mf),  'std_mean_fitness':  np.std(mf),
        'mean_converged_run_prob':   np.mean(run), 'std_converged_run_prob':   np.std(run),
        'mean_converged_short_pass_prob': np.mean(sht), 'std_converged_short_pass_prob': np.std(sht),
        'mean_converged_deep_pass_prob':  np.mean(dep), 'std_converged_deep_pass_prob':  np.std(dep),
        'mean_converged_fourth_down_aggression': np.mean(fth),
        'std_converged_fourth_down_aggression':  np.std(fth),
    }]).to_csv(os.path.join(MS_RES, 'summary_stats.csv'), index=False)

    print(f'  Saved CSVs → results/multi_seed/{RUN}/')
    return mean_best, std_best


# ---------------------------------------------------------------------------
# Best strategy verbose drives
# ---------------------------------------------------------------------------
def save_best_drives(best_genome, probs, n=10):
    strat = genome_to_strategy(best_genome)
    np.random.seed(7)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f'Best Evolved Strategy: {strat}\n')
        total = 0
        for i in range(1, n + 1):
            pts = simulate_drive(strat, drive_num=i, verbose=True, probs=probs,
                                 constraints=False, turnover_penalty=TO_PENALTY,
                                 field_position_epa=FP_EPA)
            total += pts
        print(f"\n{'='*55}")
        print(f'  {n}-drive total: {total} pts  |  avg: {total/n:.2f} pts/drive')
        print(f"{'='*55}")
    text = buf.getvalue()
    os.makedirs(R_DIR, exist_ok=True)
    with open(os.path.join(R_DIR, 'best_strategy_drives.txt'), 'w') as f:
        f.write(text)
    print(f'  Saved results/{RUN}/best_strategy_drives.txt')
    print(text)


# ---------------------------------------------------------------------------
# Figures — best single seed (figures/run5/)
# ---------------------------------------------------------------------------
def make_run5_single_figures(gen_hist):
    fit_df = pd.DataFrame({
        'generation':    gen_hist['generation'],
        'best_fitness':  gen_hist['best_fitness'],
        'mean_fitness':  gen_hist['mean_fitness'],
        'worst_fitness': gen_hist['worst_fitness'],
    })
    param_df = pd.DataFrame({k: gen_hist[k] for k in
                              ['generation', 'mean_run', 'mean_short',
                               'mean_deep', 'mean_4th']})
    make_fitness_fig(RUN, fit_df)
    make_strategy_fig(RUN, param_df)


# ---------------------------------------------------------------------------
# Multi-seed avg fitness figure (figures/multi_seed/)
# ---------------------------------------------------------------------------
def make_run5_avg_figure(mean_best, std_best):
    gens = np.arange(N_GEN + 1)
    gs_df = pd.DataFrame({'generation': gens, 'mean_best': mean_best, 'std_best': std_best})
    make_avg_fitness_fig(RUN, gs_df, n_seeds=N_SEEDS)


# ---------------------------------------------------------------------------
# Updated comparison bar chart including Run 5
# ---------------------------------------------------------------------------
def update_comparison_bar():
    ms_base = os.path.join(_BASE, 'results', 'multi_seed')
    rows = []
    run_order = [
        ('run1',  'Run 1',  'mean(pts)'),
        ('run2',  'Run 2',  'mean(pts) − 0.5·std'),
        ('run3',  'Run 3',  'mean(pts) − 0.5·std + constraints'),
        ('run4a', 'Run 4a', 'mean(pts)  [TO=−3.5]'),
        ('run4b', 'Run 4b', 'mean(pts) − 0.5·std  [TO=−3.5]'),
        ('run5',  'Run 5',  'mean(pts) [TO=−3.5 + field-pos EPA]'),
    ]
    for label, display, fn in run_order:
        csv = os.path.join(ms_base, label, 'summary_stats.csv')
        if not os.path.exists(csv):
            continue
        row = pd.read_csv(csv).iloc[0].to_dict()
        row['run']     = label
        row['display'] = display
        row['fitness_fn'] = fn
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    # overwrite all_runs_summary.csv with updated 6-run version
    summary_df.to_csv(os.path.join(ms_base, 'all_runs_summary.csv'), index=False)
    make_comparison_bar(summary_df)


# ---------------------------------------------------------------------------
# Cross-run comparison table
# ---------------------------------------------------------------------------
def print_comparison():
    RUN_META = {
        'run1':  ('mean(pts)',                              'No',  'No',  'No',  'No' ),
        'run2':  ('mean(pts) − 0.5·std',                   'No',  'Yes', 'No',  'No' ),
        'run3':  ('mean(pts) − 0.5·std',                   'No',  'Yes', 'Yes', 'No' ),
        'run4a': ('mean(pts)  [TO=−3.5]',                  'Yes', 'No',  'No',  'No' ),
        'run4b': ('mean(pts) − 0.5·std  [TO=−3.5]',        'Yes', 'Yes', 'No',  'No' ),
        'run5':  ('mean(pts)  [TO=−3.5 + FP-EPA]',         'Yes', 'No',  'No',  'Yes'),
    }
    print('\n' + '=' * 120)
    print('  FULL CROSS-RUN COMPARISON (including Run 5)')
    print('=' * 120)
    hdr = (f"  {'Run':<6}  {'Fitness Function':<40}  {'TO':>4}  {'Var':>4}  "
           f"{'Con':>4}  {'FP':>4}  {'Best':>7}  {'Mean':>7}  "
           f"{'run%':>5}  {'sht%':>5}  {'dep%':>5}  {'4th':>5}  Seed")
    print(hdr)
    print('  ' + '-' * 114)

    res_base = os.path.join(_BASE, 'results')
    for run in ('run1', 'run2', 'run3', 'run4a', 'run4b', 'run5'):
        fcsv = os.path.join(res_base, run, 'fitness_history.csv')
        pcsv = os.path.join(res_base, run, 'final_population.csv')
        scsv = os.path.join(res_base, run, 'seed.txt')
        if not os.path.exists(fcsv):
            print(f'  {run:<6}  (no data)')
            continue
        last  = pd.read_csv(fcsv).iloc[-1]
        seed  = open(scsv).read().strip() if os.path.exists(scsv) else 'unknown'
        fn, to, var, con, fp = RUN_META.get(run, ('?','?','?','?','?'))
        if os.path.exists(pcsv):
            top   = pd.read_csv(pcsv).iloc[0]
            r_pct = top['run_prob'] * 100
            s_pct = top['short_pass_prob'] * 100
            d_pct = top['deep_pass_prob'] * 100
            fth   = top['fourth_down_aggression']
        else:
            r_pct = s_pct = d_pct = fth = float('nan')
        print(f"  {run:<6}  {fn:<40}  {to:>4}  {var:>4}  {con:>4}  {fp:>4}  "
              f"{last['best_fitness']:>7.3f}  {last['mean_fitness']:>7.3f}  "
              f"{r_pct:>5.1f}  {s_pct:>5.1f}  {d_pct:>5.1f}  {fth:>5.3f}  {seed}")
    print('=' * 120)
    print('  TO=Turnover penalty  Var=Variance penalty  Con=Situational constraints  FP=Field-position EPA')
    print('=' * 120)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wall_start = time.time()

    print('=' * 80)
    print(f'  CAS Football GA — Run 5 ({N_SEEDS} seeds)')
    print(f'  f = mean(pts)  TO=−3.5  field-position EPA=True  constraints=False')
    print(f'  Pop={POP_SIZE}  Gens={N_GEN}  Drives/eval={N_DRIVES}')
    print('=' * 80)

    print('\nLoading probability distributions...')
    probs = load_probabilities()
    print('  Done.\n')

    print(f"  {'Seed':>4}  {'FinalBest':>10}  {'run%':>6}  {'sht%':>6}  {'dep%':>6}  {'4th':>6}  elapsed")
    print(f"  {'-'*65}")

    all_best_per_gen = np.empty((N_SEEDS, N_GEN + 1))
    strat_rows       = []
    all_gen_hists    = []
    best_seed_val    = -np.inf
    best_seed_data   = None

    exp_start = time.time()
    for s_idx in range(N_SEEDS):
        t0 = time.time()
        seed, gen_hist, best_per_gen, best_genome, strat_row, fitnesses = run_single(probs)

        all_best_per_gen[s_idx] = best_per_gen
        strat_row['seed_idx']   = s_idx + 1
        strat_rows.append(strat_row)
        all_gen_hists.append(gen_hist)

        if strat_row['best_fitness'] > best_seed_val:
            best_seed_val  = strat_row['best_fitness']
            best_seed_data = (gen_hist, best_genome, seed, fitnesses)

        elapsed = time.time() - t0
        m, sec  = divmod(int(elapsed), 60)
        print(
            f"  {s_idx+1:>4}  {strat_row['best_fitness']:>10.4f}  "
            f"{strat_row['converged_run_prob']*100:>6.1f}  "
            f"{strat_row['converged_short_pass_prob']*100:>6.1f}  "
            f"{strat_row['converged_deep_pass_prob']*100:>6.1f}  "
            f"{strat_row['converged_fourth_down_aggression']:>6.3f}  "
            f"{m:02d}:{sec:02d}"
        )
        sys.stdout.flush()

    print(f'\n  {N_SEEDS} seeds complete — {(time.time()-exp_start)/60:.1f} min\n')

    gen_hist_best, best_genome, best_seed, fitnesses_best = best_seed_data

    # need full population for save_best_seed — reconstruct from gen_hist final pop
    # we don't store full pop; we store the best genome only. For final_population.csv
    # we'll use the last gen data available. Re-run best seed to get full pop.
    print('--- Saving results ---')
    # Save best-seed canonical results
    # Use best seed's gen_hist; for final_population we pass fitnesses_best
    # We need the population — store it properly via re-seeding
    np.random.seed(best_seed)
    population_best = random_population(POP_SIZE)
    for _ in range(N_GEN):
        fits = evaluate_population(
            population_best, n_drives=N_DRIVES, probs=probs,
            constraints=False, variance_penalty=False,
            turnover_penalty=TO_PENALTY, field_position_epa=FP_EPA,
        )
        population_best = evolve_generation(
            population_best, fits,
            elite_n=ELITE_N, tournament_k=TOURNAMENT_K, mutation_std=MUT_STD,
        )
    fits_final = evaluate_population(
        population_best, n_drives=N_DRIVES, probs=probs,
        constraints=False, variance_penalty=False,
        turnover_penalty=TO_PENALTY, field_position_epa=FP_EPA,
    )
    save_best_seed(gen_hist_best, best_genome, population_best, fits_final, best_seed)

    print('\n--- Saving multi-seed results ---')
    mean_best, std_best = save_multi_seed(all_best_per_gen, strat_rows, all_best_per_gen)

    print('\n--- Running best strategy (10 verbose drives) ---')
    save_best_drives(best_genome, probs)

    print('\n--- Generating figures/run5/ ---')
    make_run5_single_figures(gen_hist_best)

    print('\n--- Generating figures/multi_seed/run5_avg_fitness.png ---')
    make_run5_avg_figure(mean_best, std_best)

    print('\n--- Updating multi-seed comparison bar chart ---')
    update_comparison_bar()

    print_comparison()
    print(f'\nTotal wall time: {(time.time()-wall_start)/60:.1f} min')

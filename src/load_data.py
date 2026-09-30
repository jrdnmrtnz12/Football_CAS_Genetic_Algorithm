"""
load_data.py — Pull NFL play-by-play data (2022–2024) and compute
empirical probability distributions for run, short pass, and deep pass plays.
Saves results to data/probabilities.pkl.
"""

import os
import pickle

import nfl_data_py as nfl
import numpy as np
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'probabilities.pkl')
SEASONS = [2022, 2023, 2024]


def compute_and_save_probabilities():
    print(f"Loading NFL play-by-play data for seasons {SEASONS}...")
    df = nfl.import_pbp_data(SEASONS)
    print(f"  Raw rows loaded: {len(df):,}")

    # Keep only run/pass scrimmage plays with valid yards_gained
    df = df[df['play_type'].isin(['run', 'pass'])].copy()
    df = df.dropna(subset=['yards_gained'])
    print(f"  Rows after filtering to run/pass with valid yards_gained: {len(df):,}\n")

    probs = {}

    # ------------------------------------------------------------------
    # RUN plays
    # ------------------------------------------------------------------
    runs = df[df['play_type'] == 'run'].copy()
    run_yards_dist = (
        runs['yards_gained']
        .value_counts(normalize=True)
        .sort_index()
    )
    run_fumble_rate = runs['fumble_lost'].mean()
    run_td_rate = runs['touchdown'].mean()

    probs['run'] = {
        'yards_dist': run_yards_dist,
        'fumble_rate': float(run_fumble_rate),
        'td_rate': float(run_td_rate),
    }

    print("=== RUN PLAYS ===")
    print(f"  Plays:        {len(runs):,}")
    print(f"  Fumble rate:  {run_fumble_rate:.4f}  ({run_fumble_rate*100:.2f}%)")
    print(f"  TD rate:      {run_td_rate:.4f}  ({run_td_rate*100:.2f}%)")
    print(f"  Yards gained: mean={runs['yards_gained'].mean():.2f}  "
          f"std={runs['yards_gained'].std():.2f}  "
          f"min={runs['yards_gained'].min():.0f}  "
          f"max={runs['yards_gained'].max():.0f}")
    print(f"  Dist buckets: {len(run_yards_dist)} unique yard values\n")

    # ------------------------------------------------------------------
    # SHORT PASS plays
    # ------------------------------------------------------------------
    short = df[(df['play_type'] == 'pass') & (df['pass_length'] == 'short')].copy()
    short_completion_rate = short['complete_pass'].mean()
    short_completions = short[short['complete_pass'] == 1]
    short_yards_dist = (
        short_completions['yards_gained']
        .value_counts(normalize=True)
        .sort_index()
    )
    short_int_rate = short['interception'].mean()
    short_td_rate = short['touchdown'].mean()

    probs['short_pass'] = {
        'completion_rate': float(short_completion_rate),
        'yards_dist': short_yards_dist,
        'int_rate': float(short_int_rate),
        'td_rate': float(short_td_rate),
    }

    print("=== SHORT PASS PLAYS ===")
    print(f"  Plays:           {len(short):,}  (completions: {len(short_completions):,})")
    print(f"  Completion rate: {short_completion_rate:.4f}  ({short_completion_rate*100:.2f}%)")
    print(f"  INT rate:        {short_int_rate:.4f}  ({short_int_rate*100:.2f}%)")
    print(f"  TD rate:         {short_td_rate:.4f}  ({short_td_rate*100:.2f}%)")
    print(f"  Yards (on comp): mean={short_completions['yards_gained'].mean():.2f}  "
          f"std={short_completions['yards_gained'].std():.2f}  "
          f"min={short_completions['yards_gained'].min():.0f}  "
          f"max={short_completions['yards_gained'].max():.0f}")
    print(f"  Dist buckets:    {len(short_yards_dist)} unique yard values\n")

    # ------------------------------------------------------------------
    # DEEP PASS plays
    # ------------------------------------------------------------------
    deep = df[(df['play_type'] == 'pass') & (df['pass_length'] == 'deep')].copy()
    deep_completion_rate = deep['complete_pass'].mean()
    deep_completions = deep[deep['complete_pass'] == 1]
    deep_yards_dist = (
        deep_completions['yards_gained']
        .value_counts(normalize=True)
        .sort_index()
    )
    deep_int_rate = deep['interception'].mean()
    deep_td_rate = deep['touchdown'].mean()

    probs['deep_pass'] = {
        'completion_rate': float(deep_completion_rate),
        'yards_dist': deep_yards_dist,
        'int_rate': float(deep_int_rate),
        'td_rate': float(deep_td_rate),
    }

    print("=== DEEP PASS PLAYS ===")
    print(f"  Plays:           {len(deep):,}  (completions: {len(deep_completions):,})")
    print(f"  Completion rate: {deep_completion_rate:.4f}  ({deep_completion_rate*100:.2f}%)")
    print(f"  INT rate:        {deep_int_rate:.4f}  ({deep_int_rate*100:.2f}%)")
    print(f"  TD rate:         {deep_td_rate:.4f}  ({deep_td_rate*100:.2f}%)")
    print(f"  Yards (on comp): mean={deep_completions['yards_gained'].mean():.2f}  "
          f"std={deep_completions['yards_gained'].std():.2f}  "
          f"min={deep_completions['yards_gained'].min():.0f}  "
          f"max={deep_completions['yards_gained'].max():.0f}")
    print(f"  Dist buckets:    {len(deep_yards_dist)} unique yard values\n")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_path = os.path.normpath(DATA_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(probs, f)
    print(f"Saved probability distributions to: {save_path}")

    return probs


def load_probabilities():
    """Load pre-computed distributions from disk. Always use this during simulation."""
    save_path = os.path.normpath(DATA_PATH)
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"probabilities.pkl not found at {save_path}. "
            "Run src/load_data.py first to generate it."
        )
    with open(save_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    compute_and_save_probabilities()

"""
simulator.py — Football drive simulator backed by real NFL probability distributions.

Usage:
    from simulator import Strategy, simulate_drive, evaluate_strategy

    s = Strategy(run_prob=0.4, short_pass_prob=0.4, deep_pass_prob=0.2,
                 fourth_down_aggression=0.3)
    points = simulate_drive(s, drive_num=1, verbose=True)
    fitness = evaluate_strategy(s, n_drives=100)   # mean pts/drive
"""

import os
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
_G  = '\033[92m'   # GREEN  — positive play (5+ yards, first down, TD)
_Y  = '\033[93m'   # YELLOW — neutral (1-4 yards, incomplete)
_R  = '\033[91m'   # RED    — negative (loss, INT, fumble)
_B  = '\033[94m'   # BLUE   — drive summary
_BD = '\033[1m'    # BOLD   — TD / turnover events
_X  = '\033[0m'    # RESET

# ---------------------------------------------------------------------------
# Probability cache (loaded once from disk)
# ---------------------------------------------------------------------------
_PROBS = None

def _get_probs():
    global _PROBS
    if _PROBS is None:
        pkl = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'probabilities.pkl')
        )
        if not os.path.exists(pkl):
            raise FileNotFoundError(
                f"probabilities.pkl not found at {pkl}. "
                "Run  python src/load_data.py  first."
            )
        with open(pkl, 'rb') as f:
            _PROBS = pickle.load(f)
    return _PROBS


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class Strategy:
    """Offensive strategy genome."""

    def __init__(self, run_prob: float, short_pass_prob: float,
                 deep_pass_prob: float, fourth_down_aggression: float):
        total = run_prob + short_pass_prob + deep_pass_prob
        self.run_prob              = run_prob / total
        self.short_pass_prob       = short_pass_prob / total
        self.deep_pass_prob        = deep_pass_prob / total
        self.fourth_down_aggression = float(fourth_down_aggression)

    def __repr__(self):
        return (f"Strategy(run={self.run_prob:.3f}, short={self.short_pass_prob:.3f}, "
                f"deep={self.deep_pass_prob:.3f}, 4th_aggr={self.fourth_down_aggression:.3f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pos(yardline: int) -> str:
    """Human-readable field position: OWN 25, OPP 10, etc."""
    if yardline <= 50:
        return f"OWN {yardline:2d}"
    return f"OPP {100 - yardline:2d}"


def _dn(down: int) -> str:
    return ('1st', '2nd', '3rd', '4th')[down - 1]


def _fmt(text: str, yards=None, first_down=False, event=None) -> str:
    """Apply ANSI color to a play-line string."""
    if event == 'td':
        return f"{_BD}{_G}{text}{_X}"
    if event in ('int', 'fumble'):
        return f"{_BD}{_R}{text}{_X}"
    if event == 'summary':
        return f"{_B}{text}{_X}"
    if first_down:
        return f"{_G}{text}{_X}"
    if yards is None:
        return text
    if yards >= 5:
        return f"{_G}{text}{_X}"
    if yards >= 1:
        return f"{_Y}{text}{_X}"
    if yards == 0:
        return f"{_Y}{text}{_X}"  # incomplete or no gain
    return f"{_R}{text}{_X}"     # loss of yards


# ---------------------------------------------------------------------------
# Core drive simulator
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Field-position EPA lookup (Run 5)
# ---------------------------------------------------------------------------
def _field_epa(yardline: int) -> tuple:
    """
    Return (punt_epa, fail_4th_epa) for the given yardline.

    Yardline encoding: 0 = own end zone, 100 = opponent end zone.
      own  1-20  → yardline  1-20
      own 21-40  → yardline 21-40
      own 41-50  → yardline 41-50  (includes midfield)
      opp 41-50  → yardline 51-59
      opp 21-40  → yardline 60-79
      opp  1-20  → yardline 80-99
    """
    if yardline <= 20: return -1.0, -2.0   # own  1-20
    if yardline <= 40: return -0.5, -1.5   # own 21-40
    if yardline <= 50: return -0.2, -1.0   # own 41-50
    if yardline <= 59: return  0.0, -0.5   # opp 41-50
    if yardline <= 79: return  0.0, -0.3   # opp 21-40
    return                     0.0, -0.1   # opp  1-20


def simulate_drive(strategy: Strategy, drive_num: int = 1,
                   verbose: bool = True, start_yardline: int = 25,
                   probs: dict = None, constraints: bool = True,
                   turnover_penalty: float = 0.0,
                   field_position_epa: bool = False) -> float:
    """
    Simulate one offensive drive.

    Parameters
    ----------
    constraints        : if False, skip situational play-selection constraints.
    turnover_penalty   : points applied on INT or fumble (0.0 or −3.5).
    field_position_epa : if True, punts and failed 4th-down conversions return
                         a field-position-adjusted EPA value instead of 0
                         (Run 5 only).

    Returns
    -------
    float : points scored for this drive (may be negative)
    """
    if probs is None:
        probs = _get_probs()

    run_d   = probs['run']
    short_d = probs['short_pass']
    deep_d  = probs['deep_pass']

    down        = 1
    yards_to_go = 10
    yardline    = start_yardline

    def log(text):
        if verbose:
            print(text)

    if verbose:
        hdr = (f"===== DRIVE {drive_num} | Strategy: "
               f"Run {strategy.run_prob*100:.0f}% | "
               f"Short {strategy.short_pass_prob*100:.0f}% | "
               f"Deep {strategy.deep_pass_prob*100:.0f}% | "
               f"4th Aggr: {strategy.fourth_down_aggression:.2f} =====")
        print(f"\n{_BD}{hdr}{_X}")

    MAX_PLAYS = 50  # safety cap to prevent runaway drives

    for _ in range(MAX_PLAYS):

        prefix = f"  {_dn(down)} & {yards_to_go:>2} at {_pos(yardline)}  →  "

        # ----------------------------------------------------------------
        # 4th-down decision (before running a play)
        # ----------------------------------------------------------------
        if down == 4:
            if yardline >= 67:
                # Within ~33 yards: attempt field goal
                log(_fmt(f"{prefix}FIELD GOAL  ✓", event='td'))
                log(_fmt(f"  Drive Result: 3 pts | Ended: Field Goal", event='summary'))
                return 3

            go_for_it = (yards_to_go <= 3) or (np.random.random() < strategy.fourth_down_aggression)
            if not go_for_it:
                punt_epa = _field_epa(yardline)[0] if field_position_epa else 0.0
                log(_fmt(f"{prefix}PUNT", yards=0))
                log(_fmt(f"  Drive Result: {punt_epa:.2f} pts | Ended: Punt", event='summary'))
                return punt_epa
            # Otherwise fall through and run a scrimmage play

        # ----------------------------------------------------------------
        # Select play type — apply situational constraints before sampling.
        #
        # C1  3rd/4th & short (≤3 yd):  cap deep at 20%  (short-yardage)
        # C2  Own territory (yardline<20): cap deep at 30%  (INT risk)
        # C3  Long down (yards_to_go>15): boost short by +0.20, renorm
        #                                 (checkdown value on long downs)
        #
        # Constraints are applied sequentially; each redistributes
        # within the already-adjusted weights so they compose correctly.
        # ----------------------------------------------------------------
        run_p   = strategy.run_prob
        short_p = strategy.short_pass_prob
        deep_p  = strategy.deep_pass_prob

        if constraints:
            # C1 — short-yardage: cap deep at 20%
            if down in (3, 4) and yards_to_go <= 3 and deep_p > 0.20:
                excess = deep_p - 0.20
                deep_p = 0.20
                rs = run_p + short_p
                if rs > 1e-9:
                    run_p   += excess * (run_p   / rs)
                    short_p += excess * (short_p / rs)
                else:
                    run_p = short_p = 0.40

            # C2 — own territory: cap deep at 30% to reflect INT danger
            if yardline < 20 and deep_p > 0.30:
                excess = deep_p - 0.30
                deep_p = 0.30
                rs = run_p + short_p
                if rs > 1e-9:
                    run_p   += excess * (run_p   / rs)
                    short_p += excess * (short_p / rs)
                else:
                    run_p = short_p = 0.35

            # C3 — long down: boost short pass by 0.20, then renormalize
            if yards_to_go > 15:
                short_p += 0.20
                total    = run_p + short_p + deep_p
                run_p   /= total
                short_p /= total
                deep_p  /= total

        play_type = np.random.choice(
            ['run', 'short_pass', 'deep_pass'],
            p=[run_p, short_p, deep_p]
        )

        # ----------------------------------------------------------------
        # Simulate play outcome
        # ----------------------------------------------------------------
        yards   = 0
        label   = ''
        turnover = False

        if play_type == 'run':
            if np.random.random() < run_d['fumble_rate']:
                log(_fmt(f"{prefix}RUN: FUMBLE LOST  ✗ TURNOVER", event='fumble'))
                log(_fmt(f"  Drive Result: {turnover_penalty:.1f} pts | Ended: Fumble", event='summary'))
                return turnover_penalty
            yards = int(np.random.choice(
                run_d['yards_dist'].index.values,
                p=run_d['yards_dist'].values
            ))
            sign  = '+' if yards >= 0 else ''
            label = f"RUN: {sign}{yards} yds"

        elif play_type == 'short_pass':
            r = np.random.random()
            if r < short_d['int_rate']:
                log(_fmt(f"{prefix}SHORT PASS: INTERCEPTION  ✗ TURNOVER", event='int'))
                log(_fmt(f"  Drive Result: {turnover_penalty:.1f} pts | Ended: Interception", event='summary'))
                return turnover_penalty
            elif r < short_d['int_rate'] + short_d['completion_rate']:
                yards = int(np.random.choice(
                    short_d['yards_dist'].index.values,
                    p=short_d['yards_dist'].values
                ))
                sign  = '+' if yards >= 0 else ''
                label = f"SHORT PASS: Complete, {sign}{yards} yds"
            else:
                yards = 0
                label = "SHORT PASS: Incomplete"

        else:  # deep_pass
            r = np.random.random()
            if r < deep_d['int_rate']:
                log(_fmt(f"{prefix}DEEP PASS: INTERCEPTION  ✗ TURNOVER", event='int'))
                log(_fmt(f"  Drive Result: {turnover_penalty:.1f} pts | Ended: Interception", event='summary'))
                return turnover_penalty
            elif r < deep_d['int_rate'] + deep_d['completion_rate']:
                yards = int(np.random.choice(
                    deep_d['yards_dist'].index.values,
                    p=deep_d['yards_dist'].values
                ))
                sign  = '+' if yards >= 0 else ''
                label = f"DEEP PASS: Complete, {sign}{yards} yds"
            else:
                yards = 0
                label = "DEEP PASS: Incomplete"

        # ----------------------------------------------------------------
        # Advance yardline
        # ----------------------------------------------------------------
        new_yardline = yardline + yards

        # Pushed back into own end zone
        if new_yardline <= 0:
            log(_fmt(f"{prefix}{label}", yards=yards))
            log(_fmt(f"  Drive Result: 0 pts | Ended: Backed into end zone", event='summary'))
            return 0

        # Touchdown
        if new_yardline >= 100:
            log(_fmt(f"{prefix}{label}  ★ TOUCHDOWN", event='td'))
            log(_fmt(f"  Drive Result: 7 pts | Ended: Touchdown", event='summary'))
            return 7

        yardline = new_yardline

        # ----------------------------------------------------------------
        # Update down & distance
        # ----------------------------------------------------------------
        if yards >= yards_to_go:
            # First down achieved
            log(_fmt(f"{prefix}{label}  ✓ FIRST DOWN", yards=yards, first_down=True))
            down        = 1
            yards_to_go = 10
        else:
            yards_to_go = max(1, yards_to_go - yards)
            down       += 1
            # Determine color by yards for non-first-down plays
            log(_fmt(f"{prefix}{label}", yards=yards))

            # Failed to convert 4th down (went for it and didn't make it)
            if down > 4:
                fail_epa = _field_epa(yardline)[1] if field_position_epa else 0.0
                log(_fmt(f"  Drive Result: {fail_epa:.2f} pts | Ended: Failed 4th Down", event='summary'))
                return fail_epa

    # Safety cap
    log(_fmt(f"  Drive Result: 0 pts | Ended: Play limit reached", event='summary'))
    return 0


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------
def evaluate_strategy(strategy: Strategy, n_drives: int = 100,
                      verbose: bool = False, probs: dict = None,
                      constraints: bool = True,
                      variance_penalty: bool = True,
                      turnover_penalty: float = 0.0,
                      field_position_epa: bool = False) -> float:
    """
    Evaluate a strategy over n_drives.

    Parameters
    ----------
    constraints        : passed through to simulate_drive
    variance_penalty   : fitness = mean − 0.5·std if True, else mean
    turnover_penalty   : 0.0 or −3.5 (Run 4/5)
    field_position_epa : if True, punts/failed 4th downs use EPA lookup (Run 5)

    Returns
    -------
    float : fitness score
    """
    if probs is None:
        probs = _get_probs()
    scores = [
        simulate_drive(strategy, drive_num=i + 1, verbose=verbose,
                       probs=probs, constraints=constraints,
                       turnover_penalty=turnover_penalty,
                       field_position_epa=field_position_epa)
        for i in range(n_drives)
    ]
    mean = float(np.mean(scores))
    if variance_penalty:
        return mean - 0.5 * float(np.std(scores))
    return mean


# ---------------------------------------------------------------------------
# Quick test: 3 sample drives
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)

    s = Strategy(
        run_prob=0.4,
        short_pass_prob=0.4,
        deep_pass_prob=0.2,
        fourth_down_aggression=0.3
    )

    print(f"Testing strategy: {s}")

    total_pts = 0
    for i in range(1, 4):
        pts = simulate_drive(s, drive_num=i, verbose=True)
        total_pts += pts

    print(f"\n{'='*55}")
    print(f"  3-drive total: {total_pts} pts  |  avg: {total_pts/3:.2f} pts/drive")

    # Larger silent evaluation for a stable fitness estimate
    np.random.seed(0)
    fitness = evaluate_strategy(s, n_drives=1000, verbose=False)
    print(f"  1000-drive mean fitness: {fitness:.4f} pts/drive")
    print(f"{'='*55}")

# CAS Football GA

Evolving offensive football play-calling strategies using a genetic algorithm calibrated to real NFL play-by-play data.

**Course:** CS 523 Complex Adaptive Systems, Spring 2026
**Author:** Jordan Martinez

---

## Overview

This project investigates whether emergent offensive play-calling strategies arise when a genetic algorithm evolves strategy parameters in a stochastic football environment driven by real NFL data. Each strategy is encoded as a real-valued genome with four parameters (run probability, short pass probability, deep pass probability, and fourth down aggression) and evaluated using a custom drive simulator calibrated to over 106,000 NFL plays from the 2022–2024 seasons via the nflfastR play-by-play dataset.

Six experimental conditions were tested across 10 fixed random seeds each, progressively refining either the fitness function or the simulator's drive scoring rules to investigate how environmental pressure shapes emergent behavior.

---

## Key Findings

- **Deep-pass dominance without turnover penalty** (Runs 1, 2, 3): GA converges to ~91% deep pass regardless of variance penalties or situational constraints.
- **Strategy inversion under turnover penalty** (Runs 4a, 4b): Adding a −3.5 EPA turnover penalty inverts the emergent strategy to short-pass dominance (80–90%).
- **Bimodal landscape under combined penalties** (Run 5): Adding field-position-adjusted EPA on top of the turnover penalty produces a rugged landscape where deep and short pass strategies compete as viable peaks across seeds.
- **Universal emergent findings:** Fourth down aggression converged to ≥0.92 in every run, and run plays were nearly eliminated (3.8–5.9%) across all conditions.

---

## Project Structure

```
CAS_Football_GA/
├── CLAUDE.md                  # Project brief used with Claude Code
├── README.md                  # This file
├── data/
│   └── probabilities.pkl      # Saved NFL empirical distributions
├── src/
│   ├── load_data.py           # Pulls NFL data, computes distributions
│   ├── simulator.py           # Drive simulator with play-by-play log
│   ├── genetic_algorithm.py   # GA: selection, crossover, mutation, elitism
│   └── run_experiment.py      # Main experiment runner
├── results/
│   ├── run1/                  # Baseline: raw mean fitness
│   ├── run2/                  # Variance penalty
│   ├── run3/                  # Variance penalty + situational constraints
│   ├── run4a/                 # Turnover penalty, raw fitness
│   ├── run4b/                 # Turnover penalty + variance penalty
│   ├── run5/                  # Turnover penalty + field-position EPA
│   └── multi_seed/            # 10-seed multi-run outputs
├── figures/
│   └── (per-run figures + multi-seed comparison plots)
└── venv/                      # Python 3.11 virtual environment
```

---

## Requirements

- Python 3.11
- nfl_data_py
- numpy
- pandas
- matplotlib

---

## Setup

Clone the project and enter the directory:

```bash
cd CAS_Football_GA
```

Create and activate a virtual environment using Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install nfl_data_py numpy pandas matplotlib
```

---

## Usage

### Step 1 — Load NFL data and compute probability distributions

```bash
python src/load_data.py
```

This pulls play-by-play data for the 2022–2024 NFL seasons and saves empirical distributions to `data/probabilities.pkl`.

### Step 2 — Run an experiment

```bash
python src/run_experiment.py --run run1
```

Available runs: `run1`, `run2`, `run3`, `run4a`, `run4b`, `run5`

Outputs (fitness histories, converged strategies, and figures) are saved to `results/<run>/` and `figures/<run>/`.

### Step 3 — Multi-seed runs

To reproduce the paper's results across 10 fixed seeds:

```bash
python src/run_experiment.py --run run1 --seeds 10 --master-seed 2026
```

Multi-seed outputs are saved to `results/multi_seed/<run>/` and `figures/multi_seed/<run>/`.

---

## Genome Representation

Each strategy is a real-valued genome:

- `run_prob` — probability of calling a run play
- `short_pass_prob` — probability of calling a short pass (<20 yards through the air)
- `deep_pass_prob` — probability of calling a deep pass (≥20 yards through the air)
- `fourth_down_aggression` — likelihood of going for it on 4th down vs punting/kicking

The three play probabilities are always normalized to sum to 1.0 after any genetic operation.

---

## Fitness Function Variants

| Run  | Fitness Function                              | Turnover Penalty | Additional Rules                  |
|------|-----------------------------------------------|------------------|-----------------------------------|
| Run 1 | mean(pts/drive)                              | None             | None                              |
| Run 2 | mean(pts/drive) − 0.5 × std(pts/drive)       | None             | None                              |
| Run 3 | mean(pts/drive) − 0.5 × std(pts/drive)       | None             | Situational play-calling caps     |
| Run 4a | mean(pts/drive)                             | −3.5 EPA         | None                              |
| Run 4b | mean(pts/drive) − 0.5 × std(pts/drive)      | −3.5 EPA         | None                              |
| Run 5 | mean(pts/drive)                              | −3.5 EPA         | Field-position-adjusted EPA       |

---

## Genetic Algorithm Parameters

- Population size: 100 strategies
- Generations: 100
- Drives per fitness evaluation: 100
- Selection: Tournament (k=5)
- Crossover: Uniform, with renormalization
- Mutation: Gaussian (mean=0, std=0.05), clipped to [0,1] and renormalized
- Elitism: Top 5 strategies preserved each generation

---

## Data Source

NFL play-by-play data (2022–2024 seasons) accessed via the [nfl_data_py](https://github.com/cooperdff/nfl_data_py) Python library, which wraps the [nflfastR](https://www.nflfastr.com/) dataset.

---

## AI Usage

Substantial portions of the code in this project were generated using [Claude Code](https://www.anthropic.com/claude-code) (Anthropic, March–May 2026), working from detailed specifications provided in `CLAUDE.md`. Significant portions of the drive simulator were authored directly by the project author. All code outputs were validated through the testing procedures described in the paper.

---

## License

Academic project — not licensed for commercial use.

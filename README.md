# Shapley-Based Responsibility in Noisy-OR Diffusion Models

This repository contains the code, notebooks, and figures accompanying a paper on responsibility attribution in probabilistic information diffusion networks.

The project models reposting and exposure as a noisy-OR cooperative game and assigns source-level responsibility using exact Shapley values. The framework is designed to capture overlapping exposures, diminishing marginal contribution, and interaction effects between sources in diffusion systems.

The repository includes:

- exact and approximate noisy-OR Shapley implementations,
- correctness and performance benchmarks,
- exploratory analysis of the FibVID dataset,
- global responsibility analysis,
- dynamic temporal responsibility analysis,
- generation of publication-quality figures.

---

# Repository structure

```
.
├── data/
│   └── data.csv
├── notebooks/
│   ├── data_explorer.ipynb
│   ├── global_responsibility.ipynb
│   ├── dynamic_responsibility.ipynb
│   ├── validity_checker.ipynb
├── src/
├── environment.yml 
└── README.md
```

---

# Data format

The notebooks expect a CSV file located at:

```text
data/data.csv
```

with the following columns:

```text
source
user
tweet_id
timestamp
weight
```

where:

| Column | Description |
|---|---|
| `source` | Original source or upstream account |
| `user` | Receiving/reposting user |
| `tweet_id` | Article or post identifier |
| `timestamp` | Interaction timestamp |
| `weight` | Exposure probability used in the noisy-OR model |

---

# Notebooks

## `validity_checker.ipynb`

This notebook validates both the correctness and computational performance of the noisy-OR Shapley implementations. It compares three algorithms:

- an exact exponential-time brute-force implementation,
- an exact quadratic-time implementation,
- a high-performance float64 implementation using NumPy and Numba.

The notebook verifies exact equality between the brute-force and quadratic implementations on small games, checks the Shapley efficiency identity, benchmarks runtime scaling, and measures approximation error between exact rational and float64 outputs. It also generates timing and error plots used in the paper.

---

## `data_explorer.ipynb`

This notebook performs exploratory analysis of the reposting dataset used throughout the project. It computes dataset summary statistics, participation distributions, temporal growth statistics, and weight-distribution diagnostics. The notebook analyses how concentrated activity is across sources and users using heavy-tail and log-log diagnostics, studies cumulative posting volume over time using linear trend analysis, and examines the empirical distribution of article weights using Gaussian fitting, skewness, kurtosis, and normality testing. All generated figures are saved to the `images/` directory.

---

## `global_responsibility.ipynb`

This notebook contains the main static responsibility analysis. The reposting dataset is converted into a directed multigraph where edges represent exposure events between sources and users. The notebook computes four source-level measures:

- Shapley responsibility (`C_resp`),
- sufficiency (`C_suff`),
- necessity (`C_nec`),
- outgoing-weight baseline (`C_weights`).

It then compares these measures using rank-agreement plots, rank correlations, top-k comparison tables, and Rank-Biased Overlap (RBO). The notebook also constructs an operational network of highly responsible sources and visualises the interaction structure between them using Graphviz-based network plots.

---

## `dynamic_responsibility.ipynb`

This notebook extends the responsibility framework into the temporal setting. For each user, exposure events are sorted chronologically and Shapley responsibility is recomputed repeatedly over growing exposure prefixes. This produces time-indexed responsibility deltas that track how source-level responsibility evolves over time. The notebook generates cumulative and percentage-based responsibility trajectories and produces stacked temporal visualisations showing how responsibility accumulates dynamically during diffusion.

---

# Outputs

Generated figures are saved to:

```text
notebooks/images/
```

Typical outputs include:

```text
source_participation.png
user_participation.png
cumulative_tweets.png
distribution_weights.png
rank_agreement_scatter.png
operational_network.png
responsibility_absolute.png
responsibility_percentage_to_date.png
float_vs_exact_timing.png
float_vs_exact_error.png
```

---

# Source code

The `src/` directory is intended to contain reusable implementations extracted from the notebooks, including:

- noisy-OR Shapley algorithms,
- graph construction utilities,
- static responsibility analysis,
- dynamic responsibility computation,
- plotting helpers,
- benchmarking utilities.

---

# Environment setup

Main dependencies include:

```text
pandas
numpy
networkx
matplotlib
scipy
gmpy2
numba
tqdm
graphviz
python-graphviz
```

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate <environment-name>
```

If Graphviz is missing:

```bash
conda install -c conda-forge graphviz python-graphviz
```

---

# Reproducibility

The repository prioritises exact reproducibility using rational arithmetic throughout the exact implementations. The notebooks verify the efficiency claim and include benchmarking against brute-force reference implementations.

Recommended notebook execution order:

```text
1. basic_examples.ipynb
2. validity_checker.ipynb
3. data_explorer.ipynb
4. global_responsibility.ipynb
5. dynamic_responsibility.ipynb
```

---

# Status

This repository contains research code accompanying an academic paper submission. The notebooks currently serve as the primary reproducibility artefacts, while reusable implementations are intended to be progressively moved into the `src/` directory.
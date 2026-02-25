# ValAI

A Valorant team composition optimizer powered by a permutation-invariant Transformer. It predicts win rates for 5-agent teams on any map using professional match data from [rib.gg](https://rib.gg).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Optimize a team

```bash
python valai.py
```

Prompts you to pick a map and optionally lock in agents, then suggests the highest win-rate compositions.

### Train the model

```bash
python valai.py train
```

Runs two-phase training (pairwise pretraining → full Transformer fine-tuning) on `comp_win_rates.csv` and saves the best checkpoint to `valai_model.pt`.

A pretrained checkpoint is included in the repo so you can run the optimizer immediately.

## How it works

1. **Data** — Agent and team composition win rates are scraped from rib.gg's pro match API.
2. **Credibility adjustment** — Win rates are Bayesian-shrunk toward a prior so low-sample agents/comps don't dominate.
3. **Pairwise pretraining** — Agent, role, and map embeddings are warmed up on pairwise synergy data (C(5,2) = 10 pairs per comp).
4. **Transformer** — A 6-token Transformer (1 map token + 5 agent tokens) with mean-pooling predicts team win rate. No positional encoding → permutation invariant.
5. **Solver** — Enumerates valid team completions (role-constrained) and ranks by predicted win rate.
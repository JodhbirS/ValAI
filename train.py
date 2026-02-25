# train.py — Two-phase training for the ValAI Transformer.
#
# Phase 1: Pairwise pretraining — warms up embeddings on pair-level synergy data.
# Phase 2: Full Transformer fine-tuning on 5-agent comps.

import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import CompDataset
from pair_dataset import PairDataset
from model import ValAITransformer
from constants import NUM_AGENTS, NUM_MAPS, NUM_ROLES, AGENT_ROLES, AGENT_TO_IDX

DEFAULTS: dict = dict(
    epochs=500,
    batch_size=64,
    lr=5e-4,
    weight_decay=5e-4,
    val_split=0.15,
    patience=80,
    warmup_epochs=10,
    d_model=128,
    nhead=8,
    layers=4,
    dropout=0.2,
    min_rounds=20,
    seed=42,
    save_path="valai_model.pt",
    pretrain_epochs=200,
    skip_pretrain=False,
)


def stratified_split(ds, val_frac: float, seed: int):
    """Split by map so every map has proportional validation examples."""
    rng = random.Random(seed)
    map_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        map_to_indices[ds.samples[i][1]].append(i)

    train_idx, val_idx = [], []
    for indices in map_to_indices.values():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_frac))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    return Subset(ds, train_idx), Subset(ds, val_idx)


# --- Phase 1: Pairwise Pretraining ---

class PairSynergyModel(nn.Module):
    """Small 3-token Transformer (MAP + 2 agents) for pretraining embeddings."""

    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        num_maps: int = NUM_MAPS,
        num_roles: int = NUM_ROLES,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.agent_emb = nn.Embedding(num_agents, d_model)
        self.role_emb = nn.Embedding(num_roles, d_model)
        self.baseline_proj = nn.Linear(1, d_model, bias=False)
        self.map_emb = nn.Embedding(num_maps, d_model)

        role_lookup = torch.zeros(num_agents, dtype=torch.long)
        for name, role_idx in AGENT_ROLES.items():
            if name in AGENT_TO_IDX:
                role_lookup[AGENT_TO_IDX[name]] = role_idx
        self.register_buffer("role_lookup", role_lookup)

        self.embed_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        nn.init.normal_(self.agent_emb.weight, std=0.02)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        nn.init.normal_(self.map_emb.weight, std=0.02)
        nn.init.normal_(self.baseline_proj.weight, std=0.02)

    def forward(self, agent_a, agent_b, map_id, individual_wrs):
        a1 = (
            self.agent_emb(agent_a)
            + self.role_emb(self.role_lookup[agent_a])
            + self.baseline_proj(individual_wrs[:, 0:1])
        )
        a2 = (
            self.agent_emb(agent_b)
            + self.role_emb(self.role_lookup[agent_b])
            + self.baseline_proj(individual_wrs[:, 1:2])
        )
        m = self.map_emb(map_id)

        x = torch.stack([m, a1, a2], dim=1)  # (B, 3, d)
        x = self.embed_norm(x)
        x = self.transformer(x)
        team_vec = x[:, 1:].mean(dim=1)
        return self.output_head(team_vec).squeeze(-1)


def pretrain_pairs(cfg: dict) -> dict:
    """Train PairSynergyModel and return pretrained embedding weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairDataset("comp_win_rates.csv", min_rounds=cfg["min_rounds"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    print(f"  Phase 1 → {len(ds)} pair-map samples, {cfg['pretrain_epochs']} epochs")

    model = PairSynergyModel(
        d_model=cfg["d_model"], nhead=cfg["nhead"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"] * 2, weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["pretrain_epochs"], eta_min=cfg["lr"] * 0.01,
    )

    for epoch in range(1, cfg["pretrain_epochs"] + 1):
        model.train()
        total_sq_err, total_n = 0.0, 0

        for a, b, map_id, wr, rounds, ind_wrs in loader:
            a, b = a.to(device), b.to(device)
            map_id, wr = map_id.to(device), wr.to(device)
            rounds, ind_wrs = rounds.to(device), ind_wrs.to(device)

            optimizer.zero_grad()
            pred = model(a, b, map_id, ind_wrs)
            loss = ((rounds / rounds.sum()) * (pred - wr) ** 2).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_sq_err += ((pred.detach() - wr) ** 2).sum().item()
            total_n += len(wr)

        scheduler.step()

        if epoch % 40 == 0 or epoch == 1:
            rmse = ((total_sq_err / total_n) ** 0.5) * 100
            print(f"  [pair-pretrain] Epoch {epoch:4d}/{cfg['pretrain_epochs']}  RMSE: {rmse:.2f}%")

    return {
        "agent_emb.weight": model.agent_emb.weight.data.cpu(),
        "role_emb.weight": model.role_emb.weight.data.cpu(),
        "map_emb.weight": model.map_emb.weight.data.cpu(),
        "baseline_proj.weight": model.baseline_proj.weight.data.cpu(),
    }


# --- Phase 2: Full Transformer Training ---

def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Phase 1
    pretrained_weights = None
    if not cfg.get("skip_pretrain", False):
        print("\n═══ Phase 1: Pairwise Embedding Pretraining ═══")
        pretrained_weights = pretrain_pairs(cfg)
        print("  ✓ Embeddings pretrained\n")
    else:
        print("  Skipping pairwise pretraining (--skip_pretrain)\n")

    # Dataset
    print("═══ Phase 2: Full Transformer Training ═══")
    ds = CompDataset("comp_win_rates.csv", min_rounds=cfg["min_rounds"])
    train_ds, val_ds = stratified_split(ds, cfg["val_split"], cfg["seed"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    print(f"Dataset  → train: {len(train_ds)}  val: {len(val_ds)}  (stratified by map)")

    # Model
    model = ValAITransformer(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
    ).to(device)

    if pretrained_weights is not None:
        state = model.state_dict()
        loaded = []
        for key, tensor in pretrained_weights.items():
            if key in state and state[key].shape == tensor.shape:
                state[key] = tensor
                loaded.append(key)
        model.load_state_dict(state)
        print(f"  Loaded pretrained: {', '.join(loaded)}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # LR schedule: linear warmup → cosine annealing
    warmup_epochs = cfg.get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg["epochs"] - warmup_epochs), eta_min=cfg["lr"] * 0.01,
            ),
        ],
        milestones=[warmup_epochs],
    )

    # Training loop
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_sq_err, train_n = 0.0, 0

        for agent_ids, map_id, wr, rounds, individual_wrs in train_loader:
            agent_ids = agent_ids.to(device)
            map_id, wr = map_id.to(device), wr.to(device)
            rounds, individual_wrs = rounds.to(device), individual_wrs.to(device)

            optimizer.zero_grad()
            pred = model(agent_ids, map_id, individual_wrs)
            loss = ((rounds / rounds.sum()) * (pred - wr) ** 2).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_sq_err += ((pred.detach() - wr) ** 2).sum().item()
            train_n += len(wr)

        # Validate
        model.eval()
        val_sq_err, val_n = 0.0, 0
        with torch.no_grad():
            for agent_ids, map_id, wr, rounds, individual_wrs in val_loader:
                agent_ids = agent_ids.to(device)
                map_id, wr = map_id.to(device), wr.to(device)
                individual_wrs = individual_wrs.to(device)
                pred = model(agent_ids, map_id, individual_wrs)
                val_sq_err += ((pred - wr) ** 2).sum().item()
                val_n += len(wr)

        train_loss = train_sq_err / train_n
        val_loss = val_sq_err / val_n
        scheduler.step()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "cfg": cfg,
            }, cfg["save_path"])
            flag = " ✓ saved"
        else:
            no_improve += 1
            flag = ""

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{cfg['epochs']}  "
                f"train RMSE: {(train_loss ** 0.5) * 100:.2f}%  "
                f"val RMSE: {(val_loss ** 0.5) * 100:.2f}%{flag}"
            )

        if no_improve >= cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg['patience']} epochs).")
            break

    print(f"\nBest val RMSE: {(best_val_loss ** 0.5) * 100:.2f}%  → saved to {cfg['save_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ValAI Synergy Transformer")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=v)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    train(vars(parser.parse_args()))

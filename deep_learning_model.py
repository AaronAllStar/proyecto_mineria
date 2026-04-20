"""
deep_learning_model.py
======================
Senior AI Engineer: Deep Learning component para Injury Prediction.

Arquitectura:
  - Input: Tensor tabular numérico (salida del preprocessing ML)
  - Multi-task:
      * Rama 1 → Regresión (días de recuperación)
      * Rama 2 → Clasificación (riesgo: Low/Medium/High/Critical)
  - Backbone: MLP con residual connections y dropout
  - Framework: PyTorch (con fallback a sklearn MLP si PyTorch no está disponible)

Outputs:
  - models/dl_model.pt
  - reports/dl_evaluation.json
"""

import os
import json
import numpy as np
from datetime import datetime

# ── PyTorch import with fallback ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[DL] PyTorch not available. Using sklearn MLP fallback.")

# ── sklearn fallback ──────────────────────────────────────────────────
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from config import Config


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
N_CLASSES = 4           # Low, Medium, High, Critical
EPOCHS     = 120
BATCH_SIZE = 64
LR         = 1e-3
DROPOUT    = 0.3
PATIENCE   = 15         # Early stopping patience


# ─────────────────────────────────────────────────────────
# PyTorch Model Architecture
# ─────────────────────────────────────────────────────────
if HAS_TORCH:

    class ResidualBlock(nn.Module):
        """Residual MLP block with LayerNorm + GELU activation."""
        def __init__(self, dim: int, dropout: float = DROPOUT):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            self.activation = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation(x + self.block(x))


    class InjuryDLModel(nn.Module):
        """
        Multi-task Deep Learning model for injury prediction.

        Architecture:
          Input → Embedding → ResidualBlocks → shared backbone
                                                   ├── Regression head  → days (float)
                                                   └── Classification head → risk class (4)
        """

        def __init__(self, input_dim: int, hidden_dim: int = 256,
                     n_residual: int = 3, n_classes: int = N_CLASSES,
                     dropout: float = DROPOUT):
            super().__init__()

            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.backbone = nn.Sequential(
                *[ResidualBlock(hidden_dim, dropout) for _ in range(n_residual)]
            )

            # Regression head
            self.reg_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 1),
                nn.ReLU(),  # Days ≥ 0
            )

            # Classification head
            self.clf_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: torch.Tensor):
            x = self.input_proj(x)
            x = self.backbone(x)
            days  = self.reg_head(x).squeeze(-1)
            logits = self.clf_head(x)
            return days, logits


# ─────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────
# InjuryDLTrainer
# ─────────────────────────────────────────────────────────
class InjuryDLTrainer:
    """
    Trains the DL model from tensors exported by InjuryPredictor.export_dl_tensors().
    Supports PyTorch (preferred) and sklearn MLP fallback.
    """

    def __init__(self):
        self.config = Config()
        self.model = None
        self.input_dim: int | None = None
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") \
                      if HAS_TORCH else None
        self.history: list[dict] = []
        self.metrics: dict = {}

    def load_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor_path = os.path.join(os.path.dirname(self.config.MODEL_PATH), "dl_tensors.npz")
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(
                f"DL tensors not found at {tensor_path}. "
                "Run InjuryPredictor.export_dl_tensors() first."
            )
        data = np.load(tensor_path)
        X, y_days, y_risk = data["X"], data["y_days"], data["y_risk"]
        print(f"[DL] Tensors loaded → X:{X.shape} | y_days:{y_days.shape} | y_risk:{y_risk.shape}")
        return X, y_days, y_risk

    # --------------------------------------------------
    # PyTorch training
    # --------------------------------------------------
    def _train_torch(self, X: np.ndarray, y_days: np.ndarray, y_risk: np.ndarray) -> dict:
        self.input_dim = X.shape[1]
        print(f"[DL] PyTorch training on {self.device} | input_dim={self.input_dim}")

        X_t      = torch.tensor(X, dtype=torch.float32)
        y_days_t = torch.tensor(y_days, dtype=torch.float32)
        y_risk_t = torch.tensor(y_risk, dtype=torch.long)

        dataset = TensorDataset(X_t, y_days_t, y_risk_t)
        n_val   = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

        self.model = InjuryDLModel(self.input_dim).to(self.device)
        optimizer  = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        reg_loss_fn = nn.HuberLoss(delta=5.0)          # robust to outlier days
        clf_loss_fn = nn.CrossEntropyLoss()
        stopper     = EarlyStopping(patience=PATIENCE)

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(1, EPOCHS + 1):
            # Train
            self.model.train()
            t_loss = 0.0
            for Xb, yd_b, yr_b in train_loader:
                Xb, yd_b, yr_b = Xb.to(self.device), yd_b.to(self.device), yr_b.to(self.device)
                optimizer.zero_grad()
                pred_days, logits = self.model(Xb)
                loss = 0.7 * reg_loss_fn(pred_days, yd_b) + 0.3 * clf_loss_fn(logits, yr_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item()
            scheduler.step()

            # Validate
            self.model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for Xb, yd_b, yr_b in val_loader:
                    Xb, yd_b, yr_b = Xb.to(self.device), yd_b.to(self.device), yr_b.to(self.device)
                    pred_days, logits = self.model(Xb)
                    v_loss += (0.7 * reg_loss_fn(pred_days, yd_b) + 0.3 * clf_loss_fn(logits, yr_b)).item()

            t_avg = t_loss / len(train_loader)
            v_avg = v_loss / len(val_loader)

            if v_avg < best_val_loss:
                best_val_loss = v_avg
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            self.history.append({"epoch": epoch, "train_loss": round(t_avg, 4),
                                  "val_loss": round(v_avg, 4)})

            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | train={t_avg:.4f} | val={v_avg:.4f}")

            if stopper.step(v_avg):
                print(f"  Early stopping at epoch {epoch}.")
                break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)

        # Final evaluation
        self.model.eval()
        all_pred_days, all_pred_risk, all_true_days, all_true_risk = [], [], [], []
        with torch.no_grad():
            for Xb, yd_b, yr_b in val_loader:
                Xb = Xb.to(self.device)
                pd_, lg_ = self.model(Xb)
                all_pred_days.extend(pd_.cpu().numpy())
                all_pred_risk.extend(lg_.argmax(dim=1).cpu().numpy())
                all_true_days.extend(yd_b.numpy())
                all_true_risk.extend(yr_b.numpy())

        metrics = {
            "MAE_days": float(mean_absolute_error(all_true_days, all_pred_days)),
            "R2_days": float(r2_score(all_true_days, all_pred_days)),
            "accuracy_risk": float(accuracy_score(all_true_risk, all_pred_risk)),
            "backend": "pytorch",
            "device": str(self.device),
            "epochs_run": len(self.history),
            "best_val_loss": round(best_val_loss, 4),
        }
        return metrics

    # --------------------------------------------------
    # sklearn MLP fallback
    # --------------------------------------------------
    def _train_sklearn(self, X: np.ndarray, y_days: np.ndarray, y_risk: np.ndarray) -> dict:
        print("[DL] Training sklearn MLP (PyTorch not available)...")
        X_tr, X_v, yd_tr, yd_v, yr_tr, yr_v = train_test_split(
            X, y_days, y_risk, test_size=0.2, random_state=42, stratify=y_risk
        )
        reg = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=200,
                           activation="relu", random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=200,
                            activation="relu", random_state=42)
        reg.fit(X_tr, yd_tr)
        clf.fit(X_tr, yr_tr)

        self.model = (reg, clf)
        mae = float(mean_absolute_error(yd_v, reg.predict(X_v)))
        r2  = float(r2_score(yd_v, reg.predict(X_v)))
        acc = float(accuracy_score(yr_v, clf.predict(X_v)))
        return {"MAE_days": mae, "R2_days": r2, "accuracy_risk": acc, "backend": "sklearn"}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def train(self) -> dict:
        X, y_days, y_risk = self.load_tensors()

        if HAS_TORCH:
            metrics = self._train_torch(X, y_days, y_risk)
        else:
            metrics = self._train_sklearn(X, y_days, y_risk)

        metrics["trained_at"] = datetime.now().isoformat()
        self.metrics = metrics

        print(f"\n[DL] ═══ RESULTS ═══")
        print(f"  MAE Days  : {metrics['MAE_days']:.2f}")
        print(f"  R² Days   : {metrics['R2_days']:.4f}")
        print(f"  Risk Acc  : {metrics['accuracy_risk']:.4f}")
        print(f"  Backend   : {metrics['backend']}")

        self._save()
        return metrics

    def predict(self, X_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (pred_days, pred_risk_class) for an array of preprocessed inputs.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if HAS_TORCH and isinstance(self.model, nn.Module):
            self.model.eval()
            X_t = torch.tensor(X_tensor, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                days, logits = self.model(X_t)
            return days.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()
        else:
            reg, clf = self.model
            return reg.predict(X_tensor), clf.predict(X_tensor)

    def _save(self):
        os.makedirs(os.path.dirname(self.config.MODEL_PATH), exist_ok=True)
        model_path = self.config.MODEL_PATH.replace("dino_model", "dl_model")

        if HAS_TORCH and isinstance(self.model, nn.Module):
            model_path = model_path.replace(".joblib", ".pt")
            torch.save({
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "history": self.history,
                "metrics": self.metrics,
            }, model_path)
        else:
            import joblib
            joblib.dump(self.model, model_path)

        # Save metrics JSON
        metrics_path = os.path.join(self.config.REPORT_OUTPUT, "dl_evaluation.json")
        os.makedirs(self.config.REPORT_OUTPUT, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"  DL model saved → {model_path}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = InjuryDLTrainer()
    results = trainer.train()
    print("\n[DL] Deep Learning training complete.")
    if HAS_TORCH:
        print(f"[DL] PyTorch {torch.__version__} | Device: {trainer.device}")

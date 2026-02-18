from typing import Tuple, List

from pygam import LinearGAM, s
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import optuna


def build_gam_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearGAM:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_params = None
    best_rmse = np.inf
    n_features = X.shape[1]
    lam_values = [0.1, 0.6, 1.0, 3.0, 10.0]
    for spline_order in [3, 4]:
        for n_splines in [10, 20, 30]:
            for lam in lam_values:
                terms = s(0, n_splines=n_splines, spline_order=spline_order)
                for i in range(1, n_features):
                    terms = terms + s(i, n_splines=n_splines, spline_order=spline_order)
                fold_rmses = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    gam = LinearGAM(terms=terms, lam=lam)
                    gam.fit(X_tr, y_tr)
                    preds = gam.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    fold_rmses.append(rmse)
                mean_rmse = np.mean(fold_rmses)
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_params = (n_splines, spline_order, lam)
    if best_params is None:
        return LinearGAM().fit(X, y)
    n_splines, spline_order, lam = best_params
    terms = s(0, n_splines=n_splines, spline_order=spline_order)
    for i in range(1, n_features):
        terms = terms + s(i, n_splines=n_splines, spline_order=spline_order)
    best_model = LinearGAM(terms=terms, lam=lam)
    best_model.fit(X, y)
    return best_model


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim: int = None,
        layers: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        verbose: bool = False,
        device: str = None,
    ):
        self.input_dim = input_dim
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
    
    def _build_model(self, input_dim: int) -> nn.Module:
        modules = []
        in_dim = input_dim
        for h in self.layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.ReLU())
            if self.dropout and self.dropout > 0:
                modules.append(nn.Dropout(self.dropout))
            in_dim = h
        modules.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*modules)
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        input_dim = self.input_dim or X.shape[1]
        self._model = self._build_model(input_dim).to(self.device)
        
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self._model.train()
        for epoch in tqdm(range(self.epochs), desc="DNN epochs", leave=False):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self._model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(dataset):.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.from_numpy(X).to(self.device)).cpu().numpy().ravel()
        return preds
    
def build_dnn_model(X_train: np.ndarray, y_train: np.ndarray) -> TorchRegressor:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        layers_choice = trial.suggest_categorical("layers", [(64, 32), (128, 64), (64, 32, 16)])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 60, 120)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        fold_rmses = []
        for train_idx, val_idx in tqdm(list(kf.split(X)), desc="DNN CV folds", leave=False):
            model = TorchRegressor(
                layers=layers_choice,
                dropout=dropout,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                weight_decay=weight_decay,
                verbose=False
            )
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_rmses.append(rmse)
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    print(f"Best DNN Params from Optuna: {best_params}")
    best_model = TorchRegressor(
        layers=best_params["layers"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        weight_decay=best_params["weight_decay"],
        verbose=False
    )
    best_model.fit(X, y)
    return best_model


def build_xgboost_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"])
        }
        fold_rmses: List[float] = []
        for train_idx, val_idx in tqdm(list(kf.split(X)), desc="XGBoost CV folds", leave=False):
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                **params
            )
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_rmses.append(rmse)
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"Best XGBoost Params from Optuna: {best_params}")
    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        **best_params
    )
    best_model.fit(X, y)
    return best_model


def build_lightgbm_model(X_train: np.ndarray, y_train: np.ndarray) -> LGBMRegressor:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0)
        }
        fold_rmses: List[float] = []
        for train_idx, val_idx in tqdm(list(kf.split(X)), desc="LightGBM CV folds", leave=False):
            model = LGBMRegressor(
                objective="regression",
                random_state=42,
                **params
            )
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_rmses.append(rmse)
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"Best LightGBM Params from Optuna: {best_params}")
    best_model = LGBMRegressor(
        objective="regression",
        random_state=42,
        **best_params
    )
    best_model.fit(X, y)
    return best_model


def build_catboost_model(X_train: np.ndarray, y_train: np.ndarray) -> CatBoostRegressor:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0)
        }
        fold_rmses: List[float] = []
        for train_idx, val_idx in tqdm(list(kf.split(X)), desc="CatBoost CV folds", leave=False):
            model = CatBoostRegressor(
                loss_function="RMSE",
                random_state=42,
                verbose=False,
                **params
            )
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_rmses.append(rmse)
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"Best CatBoost Params from Optuna: {best_params}")
    best_model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=42,
        verbose=False,
        **best_params
    )
    best_model.fit(X, y)
    return best_model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return y_pred, {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def ensemble_predict(models, X_test):
    # Simple averaging ensemble
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test))
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

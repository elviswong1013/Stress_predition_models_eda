from typing import Tuple, List

from utils import load_and_preprocess_data
from models import build_gam_model, build_dnn_model, build_xgboost_model, build_lightgbm_model, build_catboost_model, evaluate_model, ensemble_predict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
import mlflow
import mlflow.sklearn
import logging
from tqdm.auto import tqdm
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from fairlearn.metrics import MetricFrame
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger.info("--- Loading and Preprocessing Data ---")
    X_train, X_test, y_train, y_test, feature_names, preprocessor, X_train_raw, X_test_raw = load_and_preprocess_data()
    logger.info("Data loaded. Training set: %s, Test set: %s", X_train.shape, X_test.shape)
    
    mlflow.set_experiment("StressPrediction")
    os.makedirs("artifacts", exist_ok=True)
    
    logger.info("--- Training GAM Model ---")
    try:
        with mlflow.start_run(run_name="GAM"):
            gam_model = build_gam_model(X_train, y_train)
            gam_pred, gam_metrics = evaluate_model(gam_model, X_test, y_test, "GAM")
            try:
                mlflow.log_params({"gam_n_splines": gam_model.n_splines, "gam_lam": float(np.mean(gam_model.lam))})
            except Exception:
                pass
            mlflow.log_metrics(gam_metrics)
            joblib.dump(gam_model, "artifacts/gam.pkl")
            mlflow.log_artifact("artifacts/gam.pkl")
    except Exception as e:
        print(f"Error training GAM: {e}")
        gam_model = None
    
    logger.info("--- Training Deep Neural Network (MLP) ---")
    with mlflow.start_run(run_name="DNN_PyTorch"):
        dnn_model = build_dnn_model(X_train, y_train)
        dnn_pred, dnn_metrics = evaluate_model(dnn_model, X_test, y_test, "Deep Neural Network")
        try:
            mlflow.log_params({
                "dnn_layers": getattr(dnn_model, "layers", None),
                "dnn_dropout": getattr(dnn_model, "dropout", None),
                "dnn_lr": getattr(dnn_model, "lr", None),
                "dnn_epochs": getattr(dnn_model, "epochs", None),
                "dnn_batch_size": getattr(dnn_model, "batch_size", None),
                "dnn_weight_decay": getattr(dnn_model, "weight_decay", None)
            })
        except Exception:
            pass
        mlflow.log_metrics(dnn_metrics)
        try:
            joblib.dump(dnn_model, "artifacts/dnn.pkl")
            mlflow.log_artifact("artifacts/dnn.pkl")
        except Exception:
            pass
    
    logger.info("--- Training XGBoost Model ---")
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model = build_xgboost_model(X_train, y_train)
        xgb_pred, xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        try:
            mlflow.log_params(xgb_model.get_params())
        except Exception:
            pass
        mlflow.log_metrics(xgb_metrics)
        joblib.dump(xgb_model, "artifacts/xgb.pkl")
        mlflow.log_artifact("artifacts/xgb.pkl")

    logger.info("--- Training LightGBM Model ---")
    with mlflow.start_run(run_name="LightGBM"):
        lgbm_model = build_lightgbm_model(X_train, y_train)
        lgbm_pred, lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
        try:
            mlflow.log_params(lgbm_model.get_params())
        except Exception:
            pass
        mlflow.log_metrics(lgbm_metrics)
        joblib.dump(lgbm_model, "artifacts/lgbm.pkl")
        mlflow.log_artifact("artifacts/lgbm.pkl")

    logger.info("--- Training CatBoost Model ---")
    with mlflow.start_run(run_name="CatBoost"):
        cat_model = build_catboost_model(X_train, y_train)
        cat_pred, cat_metrics = evaluate_model(cat_model, X_test, y_test, "CatBoost")
        try:
            mlflow.log_params(cat_model.get_params())
        except Exception:
            pass
        mlflow.log_metrics(cat_metrics)
        joblib.dump(cat_model, "artifacts/catboost.pkl")
        mlflow.log_artifact("artifacts/catboost.pkl")

    logger.info("--- Ensemble Model ---")
    models = []
    if gam_model:
        models.append(gam_model)
    models.append(dnn_model)
    models.append(xgb_model)
    models.append(lgbm_model)
    models.append(cat_model)
    
    ensemble_predictions = ensemble_predict(models, X_test)
    
    # Evaluate Ensemble
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
    mae = mean_absolute_error(y_test, ensemble_predictions)
    r2 = r2_score(y_test, ensemble_predictions)
    
    logger.info("--- Ensemble Evaluation ---")
    logger.info("RMSE: %.4f", rmse)
    logger.info("MAE: %.4f", mae)
    logger.info("R2 Score: %.4f", r2)
    
    with mlflow.start_run(run_name="Ensemble"):
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})

    logger.info("--- Feature Interpretation: PDP and Feature Importance ---")
    run_pdp_plots(xgb_model, X_train, feature_names)
    run_feature_importance_plot(xgb_model, feature_names)

    logger.info("--- Running Data Drift and Fairness Checks ---")
    run_data_drift_check(X_train_raw, y_train, X_test_raw, y_test)
    run_fairness_check(y_test, ensemble_predictions, X_test_raw)


def run_data_drift_check(X_train_raw: pd.DataFrame, y_train, X_test_raw: pd.DataFrame, y_test) -> None:
    reference = X_train_raw.copy()
    reference["Stress_Level"] = y_train.values
    current = X_test_raw.copy()
    current["Stress_Level"] = y_test.values
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    os.makedirs("artifacts", exist_ok=True)
    drift_path = os.path.join("artifacts", "data_drift_report.html")
    result.save_html(drift_path)
    with mlflow.start_run(run_name="DataDrift"):
        mlflow.log_artifact(drift_path)


def run_fairness_check(y_test, ensemble_predictions: np.ndarray, X_test_raw: pd.DataFrame) -> None:
    gender_frame = MetricFrame(
        metrics=mean_absolute_error,
        y_true=y_test,
        y_pred=ensemble_predictions,
        sensitive_features=X_test_raw["Gender"]
    )
    age_bins = pd.cut(X_test_raw["Age"], bins=[0, 25, 35, 50, 100], labels=["<=25", "26-35", "36-50", ">50"])
    age_frame = MetricFrame(
        metrics=mean_absolute_error,
        y_true=y_test,
        y_pred=ensemble_predictions,
        sensitive_features=age_bins
    )
    logger.info("Fairness MAE by Gender: %s", gender_frame.by_group.to_dict())
    logger.info("Fairness MAE by Age group: %s", age_frame.by_group.to_dict())
    with mlflow.start_run(run_name="Fairness"):
        mlflow.log_dict({"gender_mae_by_group": gender_frame.by_group.to_dict(),
                         "age_mae_by_group": age_frame.by_group.to_dict()}, "fairness_metrics.json")


def run_pdp_plots(model, X_train: np.ndarray, feature_names: List[str]) -> None:
    os.makedirs("artifacts", exist_ok=True)
    X_train = np.asarray(X_train, dtype=np.float32)
    top_indices = list(range(min(4, X_train.shape[1])))
    for idx in top_indices:
        values = X_train[:, idx]
        xs = np.linspace(values.min(), values.max(), 20)
        ys = []
        X_temp = X_train.copy()
        for v in xs:
            X_temp[:, idx] = v
            preds = model.predict(X_temp)
            ys.append(float(np.mean(preds)))
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(feature_names[idx] if idx < len(feature_names) else f"feature_{idx}")
        plt.ylabel("Partial dependence")
        plt.title(f"PDP for {feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'}")
        file_path = os.path.join("artifacts", f"pdp_feature_{idx}.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        with mlflow.start_run(run_name="PDP", nested=True):
            mlflow.log_artifact(file_path)

def run_feature_importance_plot(model, feature_names: List[str]) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    os.makedirs("artifacts", exist_ok=True)
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices if i < len(feature_names)]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(sorted_importances)), sorted_importances[::-1])
    plt.yticks(range(len(sorted_names)), list(reversed(sorted_names)))
    plt.xlabel("Feature importance")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    file_path = os.path.join("artifacts", "xgb_feature_importance.png")
    plt.savefig(file_path)
    plt.close()
    with mlflow.start_run(run_name="FeatureImportance", nested=True):
        mlflow.log_artifact(file_path)


if __name__ == "__main__":
    main()

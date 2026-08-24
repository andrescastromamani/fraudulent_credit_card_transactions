"""End-to-end pipeline: load data, engineer features, train and evaluate models.

Usage:
    python run_pipeline.py
"""

from credit_card_fraud.config import REPORTS_DIR
import mlflow

from credit_card_fraud.config import (
    AUTOENCODER_MODEL_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MLP_MODEL_PATH,
)
from credit_card_fraud.dataset import CreditCardDataset
from credit_card_fraud.features import FeatureEngineer
from credit_card_fraud.modeling.predict import ModelEvaluator
from credit_card_fraud.modeling.train import AutoencoderModel, MLPModel

EPOCHS = 30
BATCH_SIZE = 2048
SEED = 42


def main() -> None:
    """Run the full fraud-detection pipeline."""
    dataset = CreditCardDataset()
    frame = dataset.load()

    engineer = FeatureEngineer(random_state=SEED)
    engineer.prepare(frame)
    class_weights = engineer.class_weights()
    print(f"X_train {engineer.X_train.shape} - X_test {engineer.X_test.shape}")
    print(f"Class weights: {class_weights}")

    mlp = MLPModel(input_dim=engineer.X_train.shape[1])
    mlp.build()
    mlp.train(
        engineer.X_train.to_numpy(),
        engineer.y_train.to_numpy(),
        class_weight=class_weights,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    X_train_normal = engineer.X_train[engineer.y_train == 0].to_numpy()
    autoencoder = AutoencoderModel(input_dim=engineer.X_train.shape[1])
    autoencoder.build()
    autoencoder.train(X_train_normal, epochs=EPOCHS, batch_size=BATCH_SIZE)

    X_test = engineer.X_test.to_numpy()
    y_test = engineer.y_test.to_numpy()

    predictions = {
        "DL 1: MLP Supervisado": mlp.predict(X_test).ravel(),
        "DL 2: Autoencoder": autoencoder.reconstruction_error(X_test),
    }
    evaluator = ModelEvaluator(y_test)
    thresholds = {
        "DL 1: MLP Supervisado": evaluator.optimal_threshold(
            predictions["DL 1: MLP Supervisado"]
        ),
        "DL 2: Autoencoder": evaluator.optimal_threshold(
            predictions["DL 2: Autoencoder"]
        ),
    }

    summary = evaluator.evaluate(predictions, thresholds=thresholds)

    results_path = REPORTS_DIR / "model_results.csv"
    summary.to_csv(results_path, index=False)
    figure_path = FIGURES_DIR / "pr_curves.png"
    evaluator.plot_precision_recall(
        predictions,
        output_path=figure_path,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="Evaluation_Comparativa"):
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "test_rows": len(y_test),
            }
        )
        for _, row in summary.iterrows():
            model_key = row["Modelo"].lower().replace(" ", "_").replace(":", "")
            mlflow.log_metrics(
                {
                    f"{model_key}_threshold": float(row["Threshold"]),
                    f"{model_key}_accuracy": float(row["Accuracy"]),
                    f"{model_key}_precision": float(row["Precision"]),
                    f"{model_key}_recall": float(row["Recall"]),
                    f"{model_key}_f1": float(row["F1-Score"]),
                    f"{model_key}_roc_auc": float(row["ROC-AUC"]),
                    f"{model_key}_pr_auc": float(row["PR-AUC"]),
                }
            )
        mlflow.log_artifact(str(results_path), artifact_path="evaluation")
        mlflow.log_artifact(str(figure_path), artifact_path="evaluation")
        mlflow.log_artifact(str(MLP_MODEL_PATH), artifact_path="checkpoints")
        mlflow.log_artifact(str(AUTOENCODER_MODEL_PATH), artifact_path="checkpoints")

    print("=" * 80)
    print("TABLA COMPARATIVA DE MODELOS DE DEEP LEARNING")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
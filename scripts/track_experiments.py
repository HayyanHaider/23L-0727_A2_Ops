"""Real local/remote tool labs, original-run imports, and validation-only instrumentation.

Imported historic metrics are explicitly tagged as imports, never as live callbacks.
The original champion remains fixed because the held-out test was already opened.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("DO_NOT_TRACK", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow import keras

from visionguard.tracking import MlflowEpochLogger, identity_tags
from visionguard.training import dataset, seed_everything, transfer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/23L-0727_A2_Ops/colab/training_cached"
OUT = ROOT / "outputs/full_assignment"


def log_keras(model, path="model", register=None):
    return mlflow.tensorflow.log_model(model, artifact_path=path,
                                       registered_model_name=register,
                                       pip_requirements=["tensorflow==2.21.0", "numpy==2.2.6"])


def run_toy():
    (x, y), _ = keras.datasets.fashion_mnist.load_data()
    x = x[:1280].astype("float32") / 255
    y = y[:1280]
    records = []
    for lr in [0.01, 0.001]:
        seed_everything()
        with mlflow.start_run(run_name=f"fashion-mnist-lr-{lr}") as run:
            mlflow.set_tags({**identity_tags(), "kind": "tool-lab", "dataset": "Fashion-MNIST"})
            mlflow.log_params(dict(learning_rate=lr, units=32, epochs=3, batch_size=64,
                                  training_examples=1024, validation_examples=256))
            model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                                      keras.layers.Dense(32, activation="relu"), keras.layers.Dense(10, activation="softmax")])
            model.compile(keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            h = model.fit(x[:1024], y[:1024], validation_data=(x[1024:], y[1024:]),
                          epochs=3, batch_size=64, callbacks=[MlflowEpochLogger()], verbose=2)
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], h.history["loss"], label="train")
            ax.plot([1, 2, 3], h.history["val_loss"], label="validation")
            ax.set(xlabel="Epoch", ylabel="Loss", title=f"Fashion-MNIST learning rate {lr}")
            ax.legend(); mlflow.log_figure(fig, "loss.png"); plt.close(fig)
            log_keras(model)
            records.append(dict(run_id=run.info.run_id, learning_rate=lr,
                                **{k: float(v[-1]) for k, v in h.history.items()}))
    return records


def prepare_features():
    dest = OUT / "features.npz"
    if dest.exists():
        return
    model = keras.models.load_model(SOURCE / "head/best.keras", compile=False)
    extractor = keras.Model(model.input, model.get_layer("head_dropout").input)
    values = {}
    for split in ["train", "val"]:
        manifest = ROOT / f"outputs/23L-0727_A2_Ops/data_evidence/{split}.csv"
        ds = dataset(manifest, ROOT / "data/processed", batch=16)
        values[split + "_x"] = extractor.predict(ds, verbose=1)
        values[split + "_y"] = pd.read_csv(manifest).label.to_numpy("float32")
    np.savez_compressed(dest, **values)


def run_instrumented():
    prepare_features()
    cached = np.load(OUT / "features.npz")
    initial = keras.models.load_model(SOURCE / "head/best.keras", compile=False).get_layer("p_ai").get_weights()
    configurations = [(0.001, 0.3), (0.0001, 0.3), (0.01, 0.3), (0.001, 0.0), (0.001, 0.5), (0.001, 0.7)]
    records = []
    for i, (lr, dropout) in enumerate(configurations):
        seed_everything()
        automatic = i == 1
        mlflow.tensorflow.autolog(disable=not automatic, log_models=False, log_input_examples=False,
                                   disable_for_unsupported_versions=False)
        with mlflow.start_run(run_name=f"frozen-head-{'autolog' if automatic else 'manual'}-{i}") as run:
            mlflow.set_tags({**identity_tags(), "kind": "instrumentation-demonstration",
                             "logging": "tensorflow-autolog" if automatic else "manual-callback",
                             "test_access": "none; post-test diagnostic; champion unchanged"})
            mlflow.log_params(dict(learning_rate=lr, dropout=dropout, backbone="MobileNetV2",
                                  frozen_backbone=True, augmentation=False, feature_size=1280,
                                  epochs=5, batch_size=32, initialization="original frozen-head checkpoint"))
            head = keras.Sequential([keras.Input((1280,)), keras.layers.Dropout(dropout),
                                     keras.layers.Dense(1, activation="sigmoid")])
            head.layers[-1].set_weights(initial)
            head.compile(keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
            h = head.fit(cached["train_x"], cached["train_y"],
                         validation_data=(cached["val_x"], cached["val_y"]),
                         epochs=5, batch_size=32,
                         callbacks=[] if automatic else [MlflowEpochLogger()], verbose=2)
            # Log the full equivalent classifier, not only the cached-feature head.
            model = transfer(dropout=dropout, weights=None)
            model.load_weights(SOURCE / "head/best.keras")
            model.get_layer("p_ai").set_weights(head.layers[-1].get_weights())
            log_keras(model)
            mlflow.log_dict(h.history, "history.json")
            records.append(dict(run_id=run.info.run_id, logging="autolog" if automatic else "manual",
                                learning_rate=lr, dropout=dropout,
                                **{k: float(v[-1]) for k, v in h.history.items()}))
    mlflow.tensorflow.autolog(disable=True)
    return records


def import_original():
    records = []
    common = ROOT / "outputs/23L-0727_A2_Ops/data_evidence"
    for stage in ["baseline", "head", "finetune"]:
        with mlflow.start_run(run_name="original-colab-" + stage) as run:
            mlflow.set_tags({**identity_tags(), "logging": "explicit-historical-import",
                             "original_runtime": "Colab T4, 2026-11-06", "kind": "pre-test-original-experiment"})
            mlflow.log_params(dict(stage=stage, learning_rate={"baseline": 0.001, "head": 0.001, "finetune": 0.00001}[stage], seed=42, image_size=224))
            frame = pd.read_csv(SOURCE / stage / "epochs.csv")
            for index, row in frame.iterrows():
                mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=index + 1)
            mlflow.log_artifacts(str(SOURCE / stage), "training")
            for name in ["train.csv", "val.csv", "test.csv"]:
                mlflow.log_artifact(str(common / name), "splits")
            mlflow.log_artifact(str(SOURCE / "requirements-lock.txt"))
            if stage == "baseline":
                m = json.loads((SOURCE / "evaluation/baseline/metrics.json").read_text())
                mlflow.log_metrics({"test_" + k: v for k, v in m.items() if isinstance(v, (int, float))})
                mlflow.log_artifacts(str(SOURCE / "evaluation/baseline"), "evaluation")
            records.append(dict(stage=stage, run_id=run.info.run_id))
    for row in pd.read_csv(SOURCE / "tuner/trials.csv").to_dict("records"):
        with mlflow.start_run(run_name="original-applied-trial-" + str(row["trial_id"])) as run:
            mlflow.set_tags({**identity_tags(), "logging": "explicit-historical-import",
                             "kind": "pre-test-original-tuner", "selection_metric": "val_loss"})
            mlflow.log_params({k: v for k, v in row.items() if k not in {"val_loss", "status"}})
            mlflow.log_metric("val_loss", row["val_loss"])
            mlflow.log_artifact(str(SOURCE / "tuner/trials.csv"))
            records.append(dict(trial_id=row["trial_id"], run_id=run.info.run_id, val_loss=row["val_loss"]))
    with mlflow.start_run(run_name="locked-champion-before-test") as run:
        mlflow.set_tags({**identity_tags(), "logging": "explicit-historical-import", "selection": "locked before original held-out test"})
        mlflow.log_params(json.loads((SOURCE / "tuner/best_hyperparameters.json").read_text()))
        mlflow.log_metric("val_loss", json.loads((SOURCE / "tuner/best_trial.json").read_text())["val_loss"])
        scores = json.loads((SOURCE / "evaluation/transfer/metrics.json").read_text())
        mlflow.log_metrics({"test_" + k: v for k, v in scores.items() if isinstance(v, (int, float))})
        mlflow.log_artifacts(str(SOURCE / "evaluation/transfer"), "evaluation")
        for name in ["train.csv", "val.csv", "test.csv"]:
            mlflow.log_artifact(str(common / name), "splits")
        mlflow.log_artifact(str(SOURCE / "requirements-lock.txt"))
        mlflow.log_artifact(str(SOURCE / "selection_before_test.json"))
        info = log_keras(keras.models.load_model(SOURCE / "tuner/best.keras", compile=False), register="23L-0727_A2_Ops")
        from mlflow import MlflowClient
        client = MlflowClient()
        first = info.registered_model_version
        second = mlflow.register_model(f"runs:/{run.info.run_id}/model", "23L-0727_A2_Ops")
        client.set_model_version_tag("23L-0727_A2_Ops", str(first), "purpose", "initial registration of locked checkpoint")
        client.set_model_version_tag("23L-0727_A2_Ops", second.version, "purpose", "deployment promotion; same verified checkpoint")
        client.set_registered_model_alias("23L-0727_A2_Ops", "champion", second.version)
        records.append(dict(champion_run_id=run.info.run_id, model_version=second.version))
    return records


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "features", "instrument", "import"], required=True)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="23L-0727_A2_Ops")
    parser.add_argument("--label", default="local")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.mode == "features":
        prepare_features(); return
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    result = {"toy": run_toy, "instrument": run_instrumented, "import": import_original}[args.mode]()
    (OUT / f"{args.label}_{args.mode}_runs.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

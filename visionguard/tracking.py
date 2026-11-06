"""Explicit per-epoch logging used alongside the autologging demonstration."""
import mlflow
from tensorflow import keras


class MlflowEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({key: float(value) for key, value in (logs or {}).items()}, step=epoch + 1)


def identity_tags(seed=42):
    import hashlib
    import subprocess
    from pathlib import Path
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dvc_data_version": hashlib.sha256(Path("data/raw.dvc").read_bytes()).hexdigest(),
            "seed": str(seed), "project": "23L-0727_A2_Ops"}

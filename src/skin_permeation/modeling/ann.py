from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from ..dependencies import MissingDependencyError, require_module

LOGGER = logging.getLogger(__name__)


def _tensorflow():
    return require_module("tensorflow", "Install tensorflow to train the ANN models.")


def build_baseline_ann(input_dim: int, learning_rate: float = 1e-4, dropout_rate: float = 0.2):
    tf = _tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, input_shape=(input_dim,)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mae", metrics=["mae"])
    return model


def build_improved_ann(
    input_dim: int,
    learning_rate: float = 1e-3,
    width: Iterable[int] = (256, 128, 64),
    dropout_rate: float = 0.25,
    l2_penalty: float = 1e-4,
):
    tf = _tensorflow()
    regularizer = tf.keras.regularizers.l2(l2_penalty)
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in width:
        layers.append(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mae", metrics=["mae"])
    return model


def train_ann(
    model,
    x_train,
    y_train,
    x_valid,
    y_valid,
    epochs: int,
    batch_size: int,
    patience: int,
    random_state: int,
    output_dir: Path,
):
    tf = _tensorflow()
    np.random.seed(random_state)
    tf.keras.utils.set_random_seed(random_state)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    history_frame = require_module("pandas").DataFrame(history.history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    return history


def save_ann(model, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(model.to_json(), encoding="utf-8")
    model.save(output_prefix.with_suffix(".keras"))


"""
train_cnn.py — train a compact CNN on MNIST (Kaggle Digit Recognizer) and create submission.
Usage (local):
    python src/train_cnn.py --data_dir data --epochs 10 --batch_size 128 --out submissions/submission_cnn_v1.csv --model_path models/cnn_v1.keras
"""
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
from utils import seed_everything, load_train_test, as_images, save_submission

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main(args):
    seed_everything(args.seed)
    X, y, X_test = load_train_test(args.data_dir)
    X_img = as_images(X)
    X_test_img = as_images(X_test)

    model = build_model()

    # Augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, validation_split=0.1
    )

    datagen.fit(X_img)
    train_flow = datagen.flow(X_img, y, batch_size=args.batch_size, subset="training")
    val_flow = datagen.flow(X_img, y, batch_size=args.batch_size, subset="validation")

    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")]
    model.fit(train_flow, epochs=args.epochs, validation_data=val_flow, callbacks=callbacks, verbose=2)

    # Predict test and save
    preds = np.argmax(model.predict(X_test_img, verbose=0), axis=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_submission(preds, args.out)

    # Save model if requested
    if args.model_path:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        print(f"Saved model to {args.model_path}")
    print(f"Wrote submission to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="submissions/submission_cnn_v1.csv")
    parser.add_argument("--model_path", type=str, default="models/cnn_v1.keras")
    args = parser.parse_args()
    main(args)

# src/train.py
import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


from preprocess import preprocess_data
from models import create_rnn_model, compile_model
from evaluate import (
    evaluate_model, plot_training_history,
    save_metrics_to_csv, plot_comparison
)
from utils import create_directories, TimeHistory



def reseed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def config_tag(cfg, idx=None, rep=None):
    tag = f"{cfg['architecture']}_bi{int(bool(cfg.get('bidirectional', False)))}_" \
          f"{cfg['activation']}_{cfg['optimizer']}_L{cfg['seq_length']}"
    cn = cfg.get("clipnorm", None)
    lr = cfg.get("learning_rate", 1e-3)
    tag += f"_clip{cn if cn is not None else 'none'}_lr{lr:g}"
    if idx is not None:
        tag = f"exp{idx:03d}_" + tag
    if rep is not None:
        tag += f"_r{rep}"
    return tag



def train_single_model(
    config,
    x_train, y_train, x_test, y_test,
    epochs=12, batch_size=32, patience=4,
    exp_idx=0, repeat_idx=1
):
    print("\n" + "=" * 100)
    print(
        f"Training: arch={config['architecture'].upper()} | "
        f"bi={bool(config.get('bidirectional', False))} | "
        f"act={config['activation']} | opt={config['optimizer']} | "
        f"lr={config.get('learning_rate', 0.001)} | L={config['seq_length']} | "
        f"clipnorm={config.get('clipnorm', None)}"
    )
    print("-" * 100)

    # Build + compile
    model = create_rnn_model(
        vocab_size=10000,
        embedding_dim=100,
        hidden_size=64,
        max_length=config["seq_length"],
        architecture=config["architecture"],            # 'rnn' | 'lstm' | 'gru'
        activation=config["activation"],                # 'sigmoid' | 'relu' | 'tanh'
        dropout_rate=0.4,
        bidirectional=bool(config.get("bidirectional", False)),
    )

    model = compile_model(
        model=model,
        optimizer=config["optimizer"],                  # 'adam' | 'sgd' | 'rmsprop'
        learning_rate=float(config.get("learning_rate", 1e-3)),
        clipnorm=config.get("clipnorm", None)          # None or float
    )

    # Callbacks
    time_cb = TimeHistory()
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True
    )

    # Train
    start = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[time_cb, early_stop],
        verbose=1
    )
    total_time = time.time() - start

    # Eval
    metrics = evaluate_model(model, x_test, y_test)  # expects 'accuracy' and 'f1_score'

    # Aggregate
    result = {
        **config,
        **metrics,
        "training_time": float(total_time),
        "avg_epoch_time": float(np.mean(time_cb.times)) if time_cb.times else 0.0,
        "epochs_trained": int(len(history.history.get("loss", []))),
    }

    # Save curve
    os.makedirs("results/plots", exist_ok=True)
    plot_training_history(
        history,
        save_path=f"results/plots/{config_tag(config, idx=exp_idx, rep=repeat_idx)}.png"
    )

    print("\nResults:")
    print(f"accuracy={result['accuracy']:.4f}  f1={result['f1_score']:.4f}  "
          f"avg_epoch_time={result['avg_epoch_time']:.2f}s  epochs={result['epochs_trained']}")
    return result, history, model



def build_thirty_suite():
    """
    30 total experiments:

    • 27 baseline (no grad clipping), fixed seq_len=100:
        architectures = [RNN, LSTM, BiLSTM]
        activations   = [sigmoid, relu, tanh]
        optimizers    = [adam, sgd, rmsprop]
        → 3 * 3 * 3 = 27

    • +3 stability checks: BiLSTM @ seq_len=100, activation=tanh,
        optimizers in [adam, sgd, rmsprop], clipnorm=1.0
        → 3 more

    Rationale: keeps the matrix balanced across RNN/LSTM/BiLSTM, while adding a
    small clipped BiLSTM set to catch any stability win.
    """
    L = 100
    activations = ["sigmoid", "relu", "tanh"]
    optimizers = ["adam", "sgd", "rmsprop"]

    configs = []

    # 27 baseline (no clipping)
    arch_defs = [
        ("rnn", False),   # RNN
        ("lstm", False),  # LSTM
        ("lstm", True),   # BiLSTM
    ]
    for arch, bi in arch_defs:
        for act in activations:
            for opt in optimizers:
                configs.append({
                    "architecture": arch,
                    "activation": act,
                    "optimizer": opt,
                    "seq_length": L,
                    "clipnorm": None,
                    "learning_rate": 1e-3,
                    "bidirectional": bi,
                })

    # +3 BiLSTM clipped (tanh × all opts)
    for opt in optimizers:
        configs.append({
            "architecture": "lstm",
            "activation": "tanh",
            "optimizer": opt,
            "seq_length": L,
            "clipnorm": 1.0,
            "learning_rate": 1e-3,
            "bidirectional": True,
        })

    assert len(configs) == 30, f"Expected 30 configs, got {len(configs)}"
    return configs



def run_suite(
    configurations,
    epochs=12,
    batch_size=32,
    patience=4,
    repeats=1
):
    create_directories()

    # Cache preprocessed tensors per seq length
    cache = {}
    all_results = []
    best_f1 = -1.0

    for i, cfg in enumerate(configurations, start=1):
        L = cfg["seq_length"]
        if L not in cache:
            x_train, y_train, x_test, y_test = preprocess_data(max_words=10000, max_len=L)
            cache[L] = (x_train, y_train, x_test, y_test)
        else:
            x_train, y_train, x_test, y_test = cache[L]

        # Repeats with different seeds; keep best by F1
        best_rep = None
        for rep in range(1, repeats + 1):
            reseed(42 + rep)
            try:
                result, history, model = train_single_model(
                    cfg, x_train, y_train, x_test, y_test,
                    epochs=epochs, batch_size=batch_size, patience=patience,
                    exp_idx=i, repeat_idx=rep
                )
            except Exception as e:
                print(f"✗ Error in exp {i} repeat {rep}: {e}")
                continue

            if (best_rep is None) or (result["f1_score"] > best_rep["f1_score"]):
                best_rep = result
                # Save the best model for this config
                os.makedirs("results", exist_ok=True)
                model.save(f"results/best_model_exp{i:03d}.h5")

        if best_rep is None:
            print(f"Skipping exp {i}: all repeats failed.")
            continue

        # Track global best
        if best_rep["f1_score"] > best_f1:
            best_f1 = best_rep["f1_score"]
            # Keep a global best checkpoint
            try:
                os.system(f"cp results/best_model_exp{i:03d}.h5 results/best_model.h5")
            except Exception:
                pass
            print(f"✓ New global best F1: {best_f1:.4f} (exp {i})")

        all_results.append(best_rep)

    # Save metrics & plots
    save_metrics_to_csv(all_results, "results/metrics.csv")
    df = pd.DataFrame(all_results)
    plot_comparison(df, "results/plots")

    print("\n" + "=" * 70)
    print("30-EXPERIMENT SUITE COMPLETE")
    print("=" * 70)
    if len(all_results) > 0:
        top = max(all_results, key=lambda r: r["f1_score"])
        print("\nBest Configuration (by F1):")
        for k, v in top.items():
            print(f"{k}: {v}")
        print("\nResults saved to: results/metrics.csv")
        print("Plots saved to: results/plots/")
    else:
        print("No successful runs recorded.")
    return all_results



def main():
    parser = argparse.ArgumentParser(
        description="Run a 30-experiment suite over RNN/LSTM/BiLSTM for IMDb sentiment"
    )
    # Suite runner
    parser.add_argument("--suite", action="store_true",
                        help="Run the fixed 30-experiment suite")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each config with different seeds; keep best F1 per config")

    # Optional: single run (for ad-hoc tests)
    parser.add_argument("--single", action="store_true", help="Run a single config instead of the suite")
    parser.add_argument("--model", type=str, default="lstm", choices=["rnn", "lstm", "gru"])
    parser.add_argument("--activation", type=str, default="tanh", choices=["sigmoid", "relu", "tanh"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--seq-length", type=int, default=100, choices=[25, 50, 100])
    parser.add_argument("--clipnorm", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bidirectional", action="store_true")

    args = parser.parse_args()
    create_directories()

    if args.single and not args.suite:
        # Ad-hoc single config
        cfg = {
            "architecture": args.model,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "seq_length": int(args.seq_length),
            "clipnorm": args.clipnorm,
            "learning_rate": float(args.lr),
            "bidirectional": bool(args.bidirectional),
        }
        x_train, y_train, x_test, y_test = preprocess_data(max_words=10000, max_len=cfg["seq_length"])
        reseed(42)
        res, hist, model = train_single_model(
            cfg, x_train, y_train, x_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            exp_idx=1, repeat_idx=1
        )
        os.makedirs("results", exist_ok=True)
        model.save("results/model.h5")
        print("\nModel saved to: results/model.h5")
        return

    # run the 30-experiment suite
    configs = build_thirty_suite()
    run_suite(
        configs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        repeats=args.repeats
    )


if __name__ == "__main__":
    main()

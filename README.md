

````markdown
# Reproduction Guide - MSML641test

## Repository Structure

```text
MSML641test/
├── data/                     # auto-created; IMDb is downloaded here
├── results/
│   ├── metrics.csv           # all experiment results (created after training)
│   ├── best_model.h5         # best checkpoint by F1 (if full sweep)
│   └── plots/                # comparison plots + per-run histories
└── src/
    ├── preprocess.py         # download + clean + tokenize + pad
    ├── models.py             # RNN/LSTM/BiLSTM builders
    ├── train.py              # single run + full experiment suite
    ├── evaluate.py           # metrics, reports, plots
    └── utils.py              # timers, plotting helpers, dirs, seeding
````

## Local Setup

```bash
git clone [https://github.com/tusu18/MSML641test.git](https://github.com/tusu18/MSML641test.git)
cd MSML641test

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1\. Preprocess Data

Downloads the IMDb 50k dataset, cleans text (lowercase, strip punctuation), tokenizes via NLTK (top 10k tokens), and creates padded sequences.

```bash
python src/preprocess.py --data_dir data --vocab_size 10000
```

### 2\. Run Experiments

#### Single Model Run

Run a specific configuration using the flags below.

**Syntax:**

```bash
python src/train.py \
  --model {rnn,lstm,gru} \
  --activation {sigmoid,relu,tanh} \
  --optimizer {adam,sgd,rmsprop} \
  --seq-length {25,50,100} \
  [--grad-clip] [--bidirectional]
```

**Examples:**

*BiLSTM (bidirectional) with Adam, 50 tokens, gradient clipping on:*

```bash
python src/train.py --model lstm --activation tanh --optimizer adam --seq-length 50 --grad-clip --bidirectional
```

*Vanilla RNN with ReLU + Adam, 100 tokens:*

```bash
python src/train.py --model rnn --activation relu --optimizer adam --seq-length 100
```

#### Full Experiment Sweep

Run the full suite over all architectures, activations, optimizers, and sequence lengths:

```bash
python src/train.py --full-experiment
```

## Script Descriptions

### `src/preprocess.py`

  * Downloads IMDb 50k, cleans text, and tokenizes (NLTK).
  * Keeps the top 10k tokens.
  * Pads or truncates sequences to 25/50/100 (length set later during training).
  * Saves vocab and intermediate artifacts for fast restarts.

### `src/models.py`

  * `create_rnn_model(...)` builds RNN/LSTM/BiLSTM architectures.
  * **Architecture:** Embedding 100, 2 recurrent layers, hidden size 64, dropout ≈ 0.4.
  * **Output:** Binary classification (sigmoid).

### `src/train.py`

  * Executes single runs based on CLI flags.
  * Executes `--full-experiment` sweep.
  * Tracks time per epoch, early stopping, and saves the best checkpoint by validation metric.

### `src/evaluate.py`

  * Computes test-set metrics (Accuracy, Macro-F1, Precision/Recall) and confusion matrix.
  * **Generates:**
      * Model comparison by F1 bar plot.
      * Accuracy vs Sequence Length line plot.
      * Per-run training history plots (accuracy/loss vs epoch).

## Results & Re-using Models

### Outputs

  * **`results/metrics.csv`**: All experiment results.
  * **`results/best_model.h5`**: Best checkpoint by F1 (auto-saved during sweep).
  * **`results/plots/`**: Comparison plots + per-run histories.

### Loading Saved Models

Evaluate the best model on the test set:

```bash
python src/evaluate.py --weights results/best_model.h5
```

```

### Next Step
Would you like me to generate the `requirements.txt` file to ensure the `pip install` command works immediately?
```

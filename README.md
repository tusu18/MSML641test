# Reproduce

#REPO

MSML641test/
├── data/                         # auto-created; IMDb is downloaded here
├── results/
│   ├── metrics.csv               # all experiment results (created after training)
│   ├── best_model.h5             # best checkpoint by F1 (if full sweep)
│   └── plots/                    # comparison plots + per-run histories
└── src/
    ├── preprocess.py             # download + clean + tokenize + pad
    ├── models.py                 # RNN/LSTM/BiLSTM builders
    ├── train.py                  # single run + full experiment suite
    ├── evaluate.py               # metrics, reports, plots
    └── utils.py                  # timers, plotting helpers, dirs, seeding

#LOCAL SETUP

git clone https://github.com/tusu18/MSML641test.git
cd MSML641test
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Preprocess (downloads IMDb and creates padded sequences)
python src/preprocess.py --data_dir data --vocab_size 10000

# Single run
python src/train.py --model lstm --activation tanh --optimizer rmsprop --seq-length 100

# Full sweep
python src/train.py --full-experiment

#SINGLE MODEL RUN

python src/train.py \
  --model {rnn,lstm,gru} \
  --activation {sigmoid,relu,tanh} \
  --optimizer {adam,sgd,rmsprop} \
  --seq-length {25,50,100} \
  [--grad-clip] [--bidirectional]

#EXAMPLE
# BiLSTM (bidirectional) with Adam, 50 tokens, gradient clipping on
python src/train.py --model lstm --activation tanh --optimizer adam --seq-length 50 --grad-clip --bidirectional

# Vanilla RNN with ReLU + Adam, 100 tokens
python src/train.py --model rnn --activation relu --optimizer adam --seq-length 100

#FULL EXPERIMENT
python src/train.py --full-experiment

What the scripts do

src/preprocess.py
	•	Downloads IMDb 50k, cleans (lowercase, strip punctuation), tokenizes (NLTK), keeps top 10k tokens.
	•	Pads or truncates sequences to 25/50/100 (length is set later during training).
	•	Saves vocab and intermediate artifacts for fast restarts.

src/models.py
	•	create_rnn_model(...) builds RNN/LSTM/BiLSTM with:
	•	Embedding 100, 2 recurrent layers, hidden size 64, dropout ≈ 0.4
	•	Binary output (sigmoid)

src/train.py
	•	Single run with the flags above.
	•	Full sweep (--full-experiment) over architectures, activations, optimizers, lengths, clipping.
	•	Tracks time per epoch, early stopping, best checkpoint by validation metric.

src/evaluate.py
	•	Computes test‐set metrics (Accuracy, Macro-F1, Precision/Recall), confusion matrix.
	•	Generates:
	•	Model comparison by F1 bar plot
	•	Accuracy vs Sequence Length line plot
	•	Per-run training history plots (accuracy/loss vs epoch)

  Re-using Saved Models
	•	Best from sweep: results/best_model.h5 (auto-saved when --full-experiment finds a new best F1).
	•	Evaluate a saved model on the test set:
  python src/evaluate.py --weights results/best_model.h5

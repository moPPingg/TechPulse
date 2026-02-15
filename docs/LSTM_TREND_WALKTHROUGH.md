# LSTM Trend Pipeline — Beginner’s Guide

This doc teaches `src/models/lstm_trend.py` assuming you only know basic Python and numpy. You’ll see: what sliding windows are, what the Dataset does, how the LSTM model is built, how training and evaluation work, then a full pipeline in pseudocode and three exercises.

---

## 1. Sliding windows

### The idea in plain language

You have a **table of days**: each row is one day, each column is a number (return, RSI, volume, etc.). You want to predict **tomorrow’s trend** (up or down).

The LSTM doesn’t take “one row = one sample” like a normal table. It takes **a sequence of rows** = one sample. So:

- **One sample** = the last `seq_len` days (e.g. 20 days) of features.
- **Label for that sample** = trend of the **next** day (day 21).

That’s a **sliding window**: you “slide” a window of 20 days along the timeline; each position of the window is one training example.

### Picture (with seq_len = 3)

Suppose the table has 6 days and 2 features (A, B):

```
Row:  0    1    2    3    4    5
      Day0 Day1 Day2 Day3 Day4 Day5
```

- **Window 1:** rows 0, 1, 2 → features for Day0, Day1, Day2. **Label** = trend of Day3 (next day).
- **Window 2:** rows 1, 2, 3 → Day1, Day2, Day3. **Label** = trend of Day4.
- **Window 3:** rows 2, 3, 4 → Day2, Day3, Day4. **Label** = trend of Day5.

So you get **3 samples**. In general, with `T` rows and window length `seq_len`, you get **T − seq_len − 1** samples (the “−1” because the last row has no “next day” to predict).

### What the code actually does

- **Input:** A DataFrame `full_df` with columns `feature_cols` and a column `return_col` (e.g. `return_1d`).
- **For each valid index `i`:**  
  - Take rows `i - seq_len + 1` to `i` (inclusive) from the feature matrix → one window of shape `(seq_len, n_features)`.  
  - The **target** is the **next-day return** at row `i + 1`; that return is then turned into a **trend label** (0 or 1 for binary: down/up).
- **Output:**  
  - `X_all`: shape `(n_samples, seq_len, n_features)` — many windows stacked.  
  - `y_all`: shape `(n_samples,)` — one integer label per window.

So: **sliding window = one sample is a chunk of consecutive days; the label is the trend of the day right after that chunk.**

---

## 2. Dataset

### What a “Dataset” is (PyTorch idea)

In PyTorch, a **Dataset** is an object that:

1. Knows **how many samples** it has (`__len__`).
2. Given an **index** `idx`, returns **one sample** and its label (`__getitem__(idx)`).

So the training loop doesn’t touch the big arrays directly; it says “give me batch 0”, “give me batch 1”, etc., and the Dataset returns the right slice.

### SlidingWindowDataset in this file

- **In:** Two numpy arrays:  
  - `X`: shape `(n_samples, seq_len, n_features)`.  
  - `y`: shape `(n_samples,)` — integer labels (0 or 1 for binary).
- **Out:** When you call `dataset[idx]`, you get the `idx`-th window and the `idx`-th label, as PyTorch tensors (so the GPU/optimizer can use them).

So the Dataset is just a thin wrapper: “I hold X and y; when you ask for index `idx`, I return `(X[idx], y[idx])` as tensors.”

### Why we need it

PyTorch’s **DataLoader** expects a Dataset. The DataLoader then:

- Groups several indices into a **batch** (e.g. 32 samples at a time).
- Can **shuffle** indices (only for training; we shuffle train, not val/test).
- Gives the training loop **batches** of `(X_batch, y_batch)` instead of one sample at a time.

So: **Dataset = “list” of (window, label) pairs; DataLoader = “give me batches from that list.”**

---

## 3. Model architecture

### In one sentence

**One sequence in → one vector out (hidden state) → one linear layer → one vector of scores (logits) per class.**

### Step by step

1. **Input shape:** `(batch_size, seq_len, n_features)`.  
   Example: (32, 20, 37) = 32 samples, each a 20-day window, 37 features per day.

2. **LSTM:**  
   - It reads the sequence **step by step** (day 0, then day 1, …, then day 19).  
   - Inside it keeps a **hidden state** that updates at each step (so it “remembers” what it saw).  
   - It outputs a vector at **each** step. We only keep the **last** one: the hidden state after seeing all 20 days.  
   - So after the LSTM: shape `(batch_size, hidden_size)` — one vector per sample (e.g. 64 numbers).

3. **Linear layer:**  
   - One linear (fully connected) layer: `hidden_size` → `num_classes`.  
   - So (batch_size, 64) → (batch_size, 2) for binary. Those 2 numbers are **logits** (scores for “down” and “up”), not yet probabilities.

4. **Output:**  
   - The model returns **logits** of shape `(batch_size, num_classes)`.  
   - To get **probabilities**, we apply **softmax** later (in the evaluation/prediction code).  
   - To get **predicted class**, we take **argmax** (which class has the biggest score).

So the **architecture** is:

```
input (batch, seq_len, n_features)
    → LSTM
    → take last step (batch, hidden_size)
    → Linear
    → logits (batch, num_classes)
```

No softmax inside the model; the loss function (cross-entropy) works with logits directly.

---

## 4. Training loop

### What we’re optimizing

We want the model’s **logits** to be good: when the true label is 1 (up), the logit for class 1 should be higher. We measure “how wrong” the logits are with **cross-entropy loss**: one number per batch; smaller = better.

### One epoch (one pass over the train set)

1. Set model to **training mode** (e.g. dropout on).
2. For **each batch** of (X_batch, y_batch):
   - Put the batch on the right device (CPU/GPU).
   - **Forward:** run the model on X_batch → get logits.
   - Compute **loss** = cross_entropy(logits, y_batch).
   - **Backward:** compute gradients of the loss with respect to every parameter (PyTorch does this with `loss.backward()`).
   - **Step:** update the parameters using those gradients (e.g. Adam: `optimizer.step()`).
   - Add the loss to a running total (so we can report average train loss for the epoch).
3. After all batches: average train loss = total_loss / number_of_batches.

### Validation (after each epoch)

1. Set model to **eval mode** (e.g. dropout off).
2. Do **not** update parameters; only compute loss and accuracy.
3. For each validation batch:
   - Forward → logits.
   - Loss for that batch; also: predicted class = argmax(logits), count how many equal the true label.
4. Average validation loss; accuracy = correct / total.

### Early stopping

- We keep a copy of the **model weights** whenever validation loss **improves** (gets lower).
- If validation loss does **not** improve for `early_stopping_patience` epochs in a row, we **stop** training.
- At the end we **restore** the best saved weights (so we don’t keep the last, possibly overfitted, model).

So the training loop = **repeat: train one epoch → validate → maybe save best weights → maybe stop; then load best weights.**

---

## 5. Evaluation

### What we need for metrics

We need:

- **True labels** for the test set: `y_test`.
- **Predicted labels**: which class the model chose (0 or 1).
- **Predicted probabilities** (optional but useful): probability of “up” so we can compute ROC-AUC.

### How prediction is done (predict_lstm)

1. Set model to **eval mode** and turn off gradient computation.
2. Split test set into **batches** (e.g. 256 samples at a time).
3. For each batch:
   - Forward → logits.
   - **Softmax(logits)** → probabilities (each row sums to 1).
   - Store those probabilities.
4. Concatenate all batches → one big array of probabilities.
5. **Predicted class** = argmax over classes (for each row).

So we get:

- `y_pred`: shape `(n_test,)` — 0 or 1 for each sample.
- `y_prob`: shape `(n_test, 2)` — probability of class 0 and class 1 for each sample.

### What “evaluate” does

The file reuses the **same** evaluation function as the LightGBM pipeline: `evaluate_trend(y_true, y_pred, y_prob, n_classes)`.

- **In:** True labels, predicted labels, predicted probabilities (optional), number of classes.
- **Out:** A dict with: **accuracy**, **precision**, **recall**, **f1**, **roc_auc**, **confusion_matrix**.

So the LSTM is judged with the **same** metrics as LightGBM, so you can compare the two fairly.

---

## 6. Full training pipeline in beginner pseudocode

Below is the **whole** flow from data to metrics, in simple steps. No Python syntax—just logic.

```
FUNCTION run():
    // ----- 1. Load data -----
    df = load price CSV for symbol from features_dir (optional: filter by from_date, to_date)
    IF include_news:
        news_df = load news features for symbol and date range
        df = merge df with news_df on date (left join, add prefix to news columns)
    feature_cols = list of numeric column names to use as features (exclude date, raw price/volume, return column)

    // ----- 2. Split in time -----
    Sort df by date
    train_df = first 60% of rows
    val_df   = next  20% of rows
    test_df  = last  20% of rows

    // ----- 3. Build sliding windows -----
    full = concatenate train_df, val_df, test_df in order (one big table)
    M = full[feature_cols] as a matrix, fill NaN, clip huge values
    next_return = for each row, the return of the NEXT row (shift -1); last row has no next
    X_list = []
    y_cont_list = []
    FOR i FROM (seq_len - 1) TO (length(full) - 2):
        window = M[ row (i - seq_len + 1) to row i ]   // shape (seq_len, n_features)
        append window to X_list
        append next_return[i+1] to y_cont_list
    X_all = stack X_list into one array   // shape (n_samples, seq_len, n_features)
    y_cont = array of y_cont_list
    y_all = turn y_cont into labels: if return > threshold_up then 1 else 0 (or 0/1/2 for 3-class)
    Split X_all and y_all into train / val / test by index ranges (so no shuffle, same time order)
    → X_train, y_train, X_val, y_val, X_test, y_test

    // ----- 4. Scale features -----
    Fit StandardScaler on X_train (flatten to 2D, fit, then reshape back)
    Transform X_train, X_val, X_test with that scaler (same shape in and out)

    // ----- 5. Create Dataset and DataLoader -----
    train_dataset = SlidingWindowDataset(X_train, y_train)   // wraps arrays, returns (X[i], y[i]) as tensors
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = SlidingWindowDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    // ----- 6. Create model and optimizer -----
    model = LSTMTrendClassifier(input_size=n_features, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()
    best_val_loss = infinity
    best_model_weights = None
    patience_counter = 0

    // ----- 7. Training loop -----
    FOR epoch = 1 TO epochs:
        model.train()
        train_loss_sum = 0
        FOR each batch (xb, yb) in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss value
        train_loss = train_loss_sum / number of batches
        record train_loss in history

        IF we have val_loader:
            model.eval()
            val_loss_sum = 0, correct = 0, total = 0
            FOR each batch (xb, yb) in val_loader (no gradients):
                logits = model(xb)
                val_loss_sum += criterion(logits, yb)
                pred = argmax(logits)
                correct += count where pred == yb
                total += batch size
            val_loss = val_loss_sum / number of val batches
            val_accuracy = correct / total
            record val_loss and val_accuracy in history
            IF val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy of model weights
                patience_counter = 0
            ELSE:
                patience_counter += 1
            IF patience_counter >= early_stopping_patience:
                break out of epoch loop
    Load best_model_weights into model (restore best checkpoint)

    // ----- 8. Evaluate on test -----
    model.eval()
    For test set in batches:
        logits = model(X_test_batch)
        probs = softmax(logits)
        collect all probs and pred = argmax(probs)
    y_pred = all predicted labels
    y_prob = all predicted probabilities
    metrics = evaluate_trend(y_test, y_pred, y_prob, n_classes)   // accuracy, precision, recall, f1, roc_auc, confusion_matrix

    RETURN metrics
```

So in order: load → split by time → build windows and labels → scale → Dataset/DataLoader → model + optimizer → train with validation and early stopping → restore best model → predict on test → compute metrics.

---

## 7. Three exercises

Do these with **only basic Python and numpy** (no PyTorch, no pandas unless stated). They help you reimplement the **ideas** in this file.

---

### Exercise 1: Sliding windows from a 2D array

**Goal:** Build the same “sliding window” samples we use for the LSTM, using only numpy.

- **Input:**
  - `M`: numpy array of shape `(T, F)` — T time steps, F features (e.g. 100 days × 5 features).
  - `seq_len`: integer (e.g. 5).
- **Output:**
  - `X`: array of shape `(N, seq_len, F)` where N = number of valid windows. Each row of `X` is one window: `X[k] = M[k : k+seq_len]`.
  - For simplicity, use **N = T - seq_len** (so the last window uses rows `T-seq_len` to `T-1`). No “next day” target yet; just build X.
- **Constraint:** Use a single loop (e.g. `for k in range(T - seq_len)`) and `np.stack` or a list of slices. No pandas.
- **Check:** For `M` of shape (10, 2) and `seq_len=3`, `X` should have shape (7, 3, 2), and `X[0]` should equal `M[0:3]`, `X[1]` = `M[1:4]`, etc.

---

### Exercise 2: “Next-day” labels for each window

**Goal:** From the same table, build the **target** for each window: the value of one column on the **next** row after the window ends.

- **Input:**
  - `M`: shape `(T, F)` (same as above).
  - `next_col`: an integer column index (e.g. 0 = first column = return).
  - `seq_len`: window length.
- **Output:**
  - `y`: 1D array of length `T - seq_len - 1`. For index `k`, `y[k]` = value at `M[k + seq_len, next_col]` (the row right after the window ends). So you have one fewer label than windows, because the last window has no “next” row.
- **Constraint:** Use indexing only (e.g. `M[k + seq_len, next_col]` in a loop), no pandas `shift`.
- **Check:** For T=6, seq_len=2, `y` has length 3; `y[0]` = `M[2, next_col]`, `y[1]` = `M[3, next_col]`, `y[2]` = `M[4, next_col]`.

---

### Exercise 3: Simple “accuracy” and “confusion” from two label arrays

**Goal:** Recreate the **idea** of accuracy and confusion matrix without using sklearn.

- **Input:**
  - `y_true`: 1D array of integers (0 or 1), length N.
  - `y_pred`: 1D array of integers (0 or 1), length N.
- **Output:**
  - `accuracy`: a float = (number of positions where y_true == y_pred) / N.
  - `confusion`: a 2×2 list of integers. `confusion[i][j]` = count of samples where true label was `i` and predicted label was `j`. So:
    - confusion[0][0] = true 0, pred 0
    - confusion[0][1] = true 0, pred 1
    - confusion[1][0] = true 1, pred 0
    - confusion[1][1] = true 1, pred 1
- **Constraint:** Use only a loop over indices and maybe a couple of conditionals; no sklearn.metrics.
- **Check:** For y_true = [0, 1, 1, 0], y_pred = [0, 1, 0, 0], accuracy = 0.75; confusion = [[2, 0], [1, 1]].

---

After doing these, you’ll have implemented: (1) building windows from a 2D array, (2) aligning “next-day” targets with those windows, and (3) accuracy and confusion matrix by hand. That matches the core of what `lstm_trend.py` does with data and evaluation.

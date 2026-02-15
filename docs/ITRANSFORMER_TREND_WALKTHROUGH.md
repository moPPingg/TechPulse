# iTransformer Trend — Beginner’s Guide (First-Year AI Student)

This doc teaches `src/models/itransformer_trend.py` step by step: what iTransformer is, why we “invert” time and features, a walk through every class and function, the forward pass in pseudocode, and how it compares to LSTM and a normal Transformer.

---

## 1. What is iTransformer, intuitively?

### The problem we have

You have a **window of time**: e.g. 20 days, and each day you have **several numbers** (features): return, RSI, volume, volatility, etc. So one sample is a **table**:

- **Rows** = time (day 0, day 1, …, day 19).
- **Columns** = variables (return, RSI, volume, …).

You want to predict **one thing** at the end (e.g. “will tomorrow go up or down?”). So you need to turn this 2D table into one prediction.

### Two natural ways to use a Transformer

A **Transformer** takes a **sequence of tokens** and lets each token “look at” the others (self-attention). So you must decide: **what is one token?**

- **Option A — time as tokens:**  
  One token = one **time step** (one day). So you have 20 tokens; each token is a vector that summarizes that day (e.g. all 37 features of that day projected to a vector). The Transformer then mixes **across time**: “day 5 attends to day 0, day 10, …”.  
  This is the **normal** time-series Transformer: **sequence length = time**.

- **Option B — variables as tokens (iTransformer):**  
  One token = one **variable** (one feature). So you have 37 tokens; each token is a vector that summarizes **that variable over the whole window** (e.g. the 20 values of “return” projected to a vector). The Transformer then mixes **across variables**: “return attends to volume, RSI, …”.  
  This is **iTransformer**: **sequence length = number of features**; time is *inside* each token.

### So what is iTransformer in one sentence?

**iTransformer is a Transformer that treats each *variable* (each feature) as one token, and each token is the *whole time series* of that variable in the window, projected into a vector.** So the model attends over **features**, not over **time steps**.

---

## 2. Why invert: time vs feature as tokens?

### Why not always use “time as tokens”?

- **Sequence length = time** (e.g. 20 or 60). In stock data we often have **many variables** (30–50+) and **moderate** time length. So we have 20–60 time tokens and 30–50 dimensions per token.
- Attention is **quadratic in sequence length**: 60 time steps → 60×60 attention. That’s fine, but the *meaning* of each token is “one day, all features mixed together.” So the model is mainly learning **temporal** relationships (day 3 vs day 15).

### What we might care about more

In finance, **relationships between variables** are central: e.g. “when volume spikes and RSI is high, return tends to…” So we want the model to ask: “given what **return** did over the window, and what **volume** did, and what **RSI** did, how do they combine to predict the next move?”

- If **variables are tokens**, the sequence length is **number of features** (e.g. 37). So we have 37 tokens; each token = “this variable’s story over the last 20 days.” Attention is **37×37**: each variable looks at every other variable. So the model explicitly learns **cross-variable** relationships.
- The **temporal** story is not lost: it’s **inside** each token. The linear layer `Linear(seq_len, d_model)` takes the 20 values of, say, “return” and compresses them into one `d_model`-sized vector. So that one vector already encodes “how return moved over time.”

### Why “invert” then?

- **Normal:** tokens = time steps → attention over **time**.
- **Inverted:** tokens = variables → attention over **variables** (and time is encoded inside each token).

So we **invert** the roles: we put **time in the “feature” position** of the linear layer (each variable has a 1D series of length `seq_len`) and **variables in the “sequence” position** (we have `n_channels` tokens). That’s why the code **transposes** `(B, T, C)` to `(B, C, T)`: so each of the `C` variables gets a 1D series of length `T` to be projected into one token.

**Summary:** We invert so that the Transformer’s expensive attention is used for **variable–variable** relationships, which are often more informative in multivariate time series, while the **temporal** pattern of each variable is encoded by a simple linear projection per variable.

---

## 3. Walk through every class and function

### Imports and dependencies (top of file)

- **torch, nn, Dataset, DataLoader:** PyTorch; the model and training use them. If PyTorch isn’t installed, they’re set to `None` and the code checks before use.
- **lightgbm_trend:** Data loading (e.g. `load_price_features`), splits (`time_aware_split`), labels (`continuous_to_trend_labels`), and evaluation (`evaluate_trend`). So iTransformer reuses the **same** data and metrics as LightGBM/LSTM.
- **lstm_trend:** Building sliding windows (`build_sliding_windows_from_splits`), scaling (`scale_sequences`), and the **same** `SlidingWindowDataset`. So the **input shape** to the model is identical to LSTM: `(batch, seq_len, n_features)`.

---

### Class: `InvertedEmbedding`

**What it is:** The layer that turns “time series per variable” into “one vector per variable” (one token per variable).

**Constructor `__init__(self, seq_len, n_channels, d_model)`:**

- **In:**  
  - `seq_len`: length of the time window (e.g. 20).  
  - `n_channels`: number of variables/features (e.g. 37).  
  - `d_model`: size of each token after projection (e.g. 64).
- **Out:** Nothing; it creates one linear layer: `self.proj = nn.Linear(seq_len, d_model)`. So **each** variable’s 1D series of length `seq_len` is mapped to a vector of size `d_model`. The same linear layer is applied to every variable (shared across channels).
- **Why:** So we get one vector per variable; those vectors become the tokens for the Transformer.

**Method `forward(self, x)`:**

- **In:** `x` of shape `(B, T, C)` = (batch, time steps, channels/variables). Example: (32, 20, 37).
- **Step 1:** `x = x.transpose(1, 2)` → shape `(B, C, T)`. So now the last dimension is **time** and the middle dimension is **channel**. So for each channel we have a 1D series of length T.
- **Step 2:** `return self.proj(x)`. The linear layer has input size `seq_len` (= T) and output size `d_model`. In PyTorch, `Linear` applies to the **last** dimension. So we get shape `(B, C, d_model)`: for each of the C variables we have one vector of size `d_model`.
- **Out:** Tensor of shape `(B, C, d_model)`. So we have **C tokens**, each of dimension `d_model`; the “sequence” the Transformer will see has length C.

---

### Class: `iTransformerTrendClassifier`

**What it is:** The full model: invert → Transformer over variables → pool → classify.

**Constructor `__init__(self, seq_len, n_channels, d_model, n_heads, n_layers, dim_feedforward, dropout, num_classes)`:**

- **In:** All the sizes and hyperparameters (sequence length, number of features, hidden size, heads, layers, etc.).
- **Out:** It creates:
  - `self.embed`: an `InvertedEmbedding(seq_len, n_channels, d_model)`.
  - `self.transformer`: a stack of `TransformerEncoderLayer` (standard: self-attention + feedforward), with `num_layers` layers. Each layer works on vectors of size `d_model`; the “sequence length” it sees is `n_channels`.
  - `self.head`: `nn.Linear(d_model, num_classes)` to go from the pooled vector to class logits.
- **Why:** One place to wire invert → Transformer → classifier.

**Method `forward(self, x)`:**

- **In:** `x` shape `(B, T, C)` (batch, time, channels).
- **Step 1:** `x = self.embed(x)` → `(B, C, d_model)`.
- **Step 2:** `x = self.transformer(x)` → still `(B, C, d_model)`. Each of the C tokens has been updated by self-attention (and feedforward) over the C tokens.
- **Step 3:** `x = x.mean(dim=1)` → `(B, d_model)`. We **mean-pool** over the C tokens: one vector per sample that summarizes all variables.
- **Step 4:** `return self.head(x)` → `(B, num_classes)`. Logits for each class (e.g. down/up).
- **Out:** Tensor of shape `(B, num_classes)`.

---

### Function: `train_itransformer_trend(...)`

**What it is:** The training loop: cross-entropy loss, Adam, optional validation and early stopping.

**In:**

- `model`: the `iTransformerTrendClassifier` (or any `nn.Module` with the same interface).
- `X_train`, `y_train`: numpy arrays (windows and integer labels).
- Optional `X_val`, `y_val` for validation.
- `device`, `epochs`, `batch_size`, `lr`, `early_stopping_patience`, `verbose`.

**What it does:**

1. Puts the model on `device`, creates Adam optimizer and `CrossEntropyLoss`.
2. Builds a `SlidingWindowDataset` from (X_train, y_train) and a `DataLoader` (shuffle=True for training). If val data is given, builds a val DataLoader (shuffle=False).
3. For each epoch:
   - **Train:** For each batch (xb, yb), forward pass → logits → loss → backward → optimizer step. Records average train loss.
   - **Val (if present):** No gradients; forward on val batches; computes val loss and val accuracy (fraction of correct predictions). If val loss improves, saves a copy of model weights and resets patience. If it doesn’t improve for `early_stopping_patience` epochs, breaks out of the loop.
4. After the loop, loads the best saved weights back into the model.
5. Returns a dict `history` with lists: `train_loss`, `val_loss`, `val_accuracy` per epoch.

**Out:** The `history` dict. The trained model is modified in place (and best weights are restored).

**Why:** Single place to train the iTransformer trend model so the pipeline and experiments share the same logic.

---

### Function: `predict_itransformer(model, X, device, batch_size)`

**What it is:** Run the model on a set of windows and return predicted classes and class probabilities.

**In:** Trained model, numpy array `X` of shape `(n_samples, seq_len, n_features)`, device, and batch size for inference.

**What it does:**

1. Puts model in eval mode and on the right device.
2. Splits `X` into batches. For each batch: convert to tensor → forward → softmax on logits → store probabilities. No gradients.
3. Concatenates all batch probabilities into one array.
4. Predicted class = argmax over classes (axis=1).

**Out:** Two numpy arrays: `preds` (integer labels) and `probs` (probabilities per class), same length as number of samples.

**Why:** Evaluation (and ROC-AUC) needs both labels and probabilities; this centralizes that.

---

### Class: `iTransformerTrendPipeline`

**What it is:** End-to-end pipeline: load data → split → build windows + scale → train → evaluate. Same data and splits as LSTM trend.

**Constructor `__init__(...)`:**
- **In:** Symbol, features_dir, date range, include_news, return column, trend thresholds, n_classes, seq_len, train/val/test ratios, purge_gap, and all model/training hyperparameters (d_model, n_heads, n_layers, dropout, epochs, batch_size, lr, early_stopping_patience, device).
- **Out:** Stores all of that in `self` and sets `self.df`, `self.feature_cols`, `self.X_train`, etc., to `None` (filled later).
- **Why:** One configuration object for the whole run.

**`load_data()`:** Calls `load_price_features`, optionally merges news via `load_news_daily_features` and `merge_multimodal`, then `get_feature_columns_for_trend`. Sets `self.df` and `self.feature_cols`. Returns `self.df`.

**`split()`:** If needed, calls `load_data()`; then `time_aware_split(self.df, ...)`. Returns (train_df, val_df, test_df).

**`build_sequences(train_df, val_df, test_df)`:** Calls `build_sliding_windows_from_splits` to get (X_train, y_train, X_val, y_val, X_test, y_test), then `scale_sequences` and stores results and scaler in `self`. No return.

**`train()`:** Builds `iTransformerTrendClassifier` from `self.X_train` shape (seq_len, n_feat), then calls `train_itransformer_trend(...)` with self’s data and hyperparameters. Sets `self.model` and `self.history`. Returns `self.history`.

**`evaluate()`:** Calls `predict_itransformer(self.model, self.X_test, ...)`, then `evaluate_trend(self.y_test, y_pred, y_prob, n_classes)`. Stores result in `self.metrics`. Returns `self.metrics`.

**`run()`:** Calls in order: `load_data()` → `split()` → `build_sequences(...)` → `train()` → `evaluate()`. Returns `self.metrics`.

---

## 4. Model forward pass in pseudocode

We only need the **model** part: from one batch of windows to logits. Data loading, training loop, and evaluation are the same idea as in the LSTM doc.

```
FUNCTION forward(x):
    // x shape: (B, T, C) = (batch_size, seq_len, n_channels)
    // Example: (32, 20, 37)

    // ----- Inverted embedding -----
    x = transpose(x, so that shape is (B, C, T))
    // Now: (32, 37, 20). Each of the 37 channels has a 1D series of length 20.
    x = Linear_layer(x)   // Linear has input size T, output size d_model; applied to last dim
    // After Linear: (B, C, d_model) = (32, 37, 64). So 37 tokens, each of size 64.

    // ----- Transformer encoder -----
    FOR each layer in transformer_layers:
        // Self-attention: each of the 37 tokens looks at all 37 tokens
        x = self_attention(x)   // input (B, C, d_model), output (B, C, d_model)
        x = feed_forward(x)    // (B, C, d_model) -> (B, C, d_model)
    // Still (B, C, d_model).

    // ----- Pool: one vector per sample -----
    x = mean(x, over dimension 1)   // (B, C, d_model) -> (B, d_model)
    // So we average the 37 token vectors into one vector per sample.

    // ----- Classification head -----
    logits = Linear_head(x)   // (B, d_model) -> (B, num_classes)
    RETURN logits
```

So in one line: **transpose → linear (time → d_model) → Transformer over C tokens → mean over C → linear → logits.**

---

## 5. Compare to LSTM and normal Transformer

### iTransformer vs LSTM

| Aspect | LSTM | iTransformer |
|--------|------|----------------|
| **What is the “sequence”?** | Time. The LSTM reads step by step: day 0, day 1, …, day 19. | Variables. The “sequence” is the 37 features; each “token” is that feature’s 20-day series. |
| **What does the model mix?** | Information **across time** (recurrent state + hidden state). | Information **across variables** (self-attention over the 37 tokens). |
| **Where is time?** | Time is the order the LSTM processes. | Time is **inside** each token (the 20 values projected by the first linear). |
| **Inductive bias** | Strong **temporal** bias: order matters, recurrence. | Strong **cross-variable** bias: “how do features relate?”; time is summarized per variable. |
| **Output from sequence** | Usually the **last** hidden state (after seeing all 20 steps). | **Mean** over the 37 tokens (after attention). |

So: **LSTM = “walk through time, remember; then use the last state.” iTransformer = “summarize each variable over time into one vector; then let variables attend to each other and average.”**

### iTransformer vs normal (time-step) Transformer

| Aspect | Normal time-series Transformer | iTransformer |
|--------|--------------------------------|--------------|
| **Token** | One **time step** (e.g. one day, often all features projected together). | One **variable** (one feature’s full time series projected to a vector). |
| **Sequence length** | T (e.g. 20 or 60). | C (e.g. 37 features). |
| **Attention over** | **Time**: “day 3 attends to day 0, 1, 2, 4, …”. | **Variables**: “return attends to volume, RSI, …”. |
| **Cost** | O(T²) in time length. | O(C²) in number of variables. So if C < T, iTransformer can be cheaper. |
| **What it’s good at** | Long-range **temporal** dependencies. | **Cross-variable** structure; time is encoded inside each token. |

So: **Normal Transformer on time = “treat each day as a token, attend over days.” iTransformer = “treat each feature as a token, attend over features; each token already carries that feature’s time series.”**

### When might you prefer which?

- **LSTM:** When you care a lot about **order in time** and recurrence (e.g. strict temporal dependencies, small number of features).
- **Normal Transformer (time as tokens):** When you have **long** sequences and need long-range time dependencies; you’re okay with O(T²) and many time tokens.
- **iTransformer:** When you have **many variables** and care more about **how they interact** (e.g. volume vs return, RSI vs volatility); time is important but can be summarized per variable first, then variables are mixed.

In this project, iTransformer uses the **same** sliding windows and labels as LSTM, so the only difference is **how** the window is processed: LSTM over time, iTransformer over variables (with time inside each variable’s token).

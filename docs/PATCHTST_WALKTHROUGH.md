# PatchTST — Patching, Tokens, Training, and How It Differs from LSTM and iTransformer

This doc explains `src/models/forecasting/patchtst.py`: what patching is, how patches become tokens, how training and prediction work, how PatchTST differs from LSTM and iTransformer, and the full forward pass and training logic in pseudocode.

---

## 1. What patching means

### The raw input

You have one **sequence** per sample: e.g. 20 time steps × 37 features. So one sample is a matrix of shape `(20, 37)` — 20 days, 37 numbers per day. The model must turn this into **one number** (e.g. next-day return).

### The problem with “one time step = one token”

If we treat **each time step** (each day) as one token, we have 20 tokens. The Transformer then does self-attention over 20 positions. That works, but:

- Attention cost grows with the **square** of the sequence length (20² = 400).
- Each token is “one day, all 37 features” — so we’re mixing time and features in one vector.

### Patching: group time steps into chunks

**Patching** means: instead of 20 tokens (one per day), we group consecutive **days** into **patches** and treat **one patch = one token**.

- Example: `patch_len = 4`. The 20 time steps become **5 patches**:
  - Patch 0: days 0–3 (4 steps × 37 features).
  - Patch 1: days 4–7.
  - Patch 2: days 8–11.
  - Patch 3: days 12–15.
  - Patch 4: days 16–19.
- So we have **5 tokens** instead of 20. The **sequence length** the Transformer sees is **5** (number of patches), not 20 (number of days).

So **patching** = “split the time axis into non-overlapping chunks; each chunk is one patch.” The code uses **non-overlapping** patches: `n_patches = seq_len // patch_len`, and trims the sequence so its length is exactly `n_patches * patch_len`.

**Why do it?**

- **Shorter sequence** → cheaper and often more stable attention (5² vs 20²).
- Each token can encode **local time** (e.g. “what happened in these 4 days”) via a linear layer; the Transformer then mixes **across patches** (more global time).

---

## 2. How patches become tokens

### From 3D input to patch vectors

- **Input:** `x` of shape `(B, T, C)` = (batch, seq_len, n_channels). Example: (32, 20, 37).
- We require `T` to be divisible by `patch_len` (e.g. 20 and 4). So we may trim: use only the **last** `effective_seq_len` steps (e.g. 20), and `effective_seq_len = n_patches * patch_len`.

**Step 1 — Reshape into patches:**

- Take the last `effective_seq_len` time steps: `x = x[:, -T_use:, :]` so we don’t use “future” beyond our window.
- Reshape so that the time dimension is split into patches:
  - Before: `(B, T, C)` e.g. (32, 20, 37).
  - After reshape: `(B, n_patches, patch_len, C)` e.g. (32, 5, 4, 37). So for each sample we have 5 patches, each patch is 4×37.
  - Then flatten the last two dimensions of each patch: `(B, n_patches, patch_len * C)` e.g. (32, 5, 148). So **one patch** is 148 numbers (4×37).

**Step 2 — Project to d_model:**

- Apply one linear layer: `Linear(patch_len * n_channels, d_model)`. So (32, 5, 148) → (32, 5, 64). Now each of the 5 patches is a vector of size **d_model** (e.g. 64). Those are the **tokens**.

**Step 3 — Positional embedding (optional):**

- Add a learned vector per patch position: `x = x + pos_embed`, so the model knows “first patch vs last patch.” Shape stays (B, n_patches, d_model).

So: **patch = contiguous block of (patch_len × n_channels) values → flatten → linear → one d_model-sized vector = one token.** The Transformer then sees **n_patches** tokens, each of dimension d_model.

---

## 3. How training and prediction work

### Training

- **Inputs:** `X_train` shape `(n_samples, seq_len, n_features)`, `y_train` shape `(n_samples,)` — continuous target (e.g. next-day return). Optionally `X_val`, `y_val`.
- **Scaling:** Fit `StandardScaler` on flattened `X_train` and on `y_train`; transform train (and val if present). So both inputs and targets are standardized; we inverse-transform predictions at the end.
- **Model:** `PatchTST(seq_len, n_channels, patch_len, d_model, ...)` — patch embedding + Transformer + head that outputs one scalar per sample.
- **Loop:** For each epoch, for each batch (xb, yb): forward → scalar prediction per sample → **MSE loss** vs yb → backward → optimizer step. If val is provided: compute val MSE, keep best model weights, early stop if val loss doesn’t improve for `early_stopping_patience` epochs. Finally restore best weights.
- **Output:** The trained model (and in `PatchTSTForecaster`, the scalers and `_history`). This is **regression** (predict a number), not classification.

### Prediction

- **Input:** `X` shape `(n_samples, seq_len, n_features)`. Use the **same** `seq_len` and scaling as in fit (take last `seq_len` steps, transform with `scaler_x`).
- **Forward:** Run the model in eval mode, in batches; no gradients. Model output is shape `(batch,)` — one scalar per sample.
- **Output:** Concatenate batch outputs and **inverse-transform** with `scaler_y` so predictions are back in original (e.g. return) scale. Return 1D array of length n_samples.

So: **same sliding-window format as LSTM/iTransformer; target is continuous; train with MSE; predict scalar and unscale.**

---

## 4. How PatchTST differs from LSTM and iTransformer

| Aspect | LSTM | iTransformer | PatchTST |
|--------|------|--------------|----------|
| **What is one “token” or step?** | One **time step** (one day, all features). The LSTM processes 20 steps in order. | One **variable** (one feature’s full 20-day series projected to a vector). 37 tokens. | One **patch** (e.g. 4 consecutive days × all features, flattened and projected). 5 tokens. |
| **Sequence length** | T (e.g. 20). | C (e.g. 37). | n_patches = T / patch_len (e.g. 5). |
| **What gets mixed?** | Information **across time** (recurrent + hidden state). | Information **across variables** (attention over 37 variable-tokens). | Information **across time chunks** (attention over 5 patch-tokens). |
| **Where is “local time”?** | Inside the recurrence (each step sees one day). | Inside each variable’s embedding (Linear(seq_len, d_model) per variable). | Inside each patch (patch = patch_len days; linear(patch_len*C, d_model)). |
| **Task in this repo** | Trend **classification** (0/1 or 0/1/2). | Trend **classification**. | **Regression** (next-step return); can be thresholded to trend for comparison. |
| **Output head** | Linear(hidden, num_classes) → logits. | Mean over variable-tokens → Linear(d_model, num_classes). | Mean (or last) over patch-tokens → Linear(d_model, 1) → scalar. |

**In one line each:**

- **LSTM:** “Walk through time step by step; use the last hidden state to classify.”
- **iTransformer:** “Summarize each variable over time into one vector; attend over variables; pool and classify.”
- **PatchTST:** “Summarize each time chunk (patch) into one vector; attend over patches; pool and regress.”

---

## 5. Full forward pass in pseudocode

```
FUNCTION PatchTST_forward(x):
    // x shape: (B, T, C) = (batch, seq_len, n_channels). Example: (32, 20, 37)

    // ----- Patch embedding -----
    T_use = min(T, n_patches * patch_len)   // use only valid length (e.g. 20)
    x = x[:, last T_use steps, :]           // (B, T_use, C)
    // Reshape: (B, T_use, C) -> (B, n_patches, patch_len, C)
    //   e.g. (32, 20, 37) -> (32, 5, 4, 37)
    x = reshape(x, (B, n_patches, patch_len, C))
    // Flatten each patch: (B, n_patches, patch_len * C)
    x = reshape(x, (B, n_patches, patch_len * C))   // (32, 5, 148)
    x = Linear(x)   // (B, n_patches, patch_len*C) -> (B, n_patches, d_model)
    IF use_pos_embed:
        x = x + pos_embed   // (B, n_patches, d_model)
    // Now x is (B, n_patches, d_model): n_patches tokens, each of size d_model.

    // ----- Transformer encoder -----
    FOR each layer in transformer_layers:
        x = self_attention(x)   // (B, n_patches, d_model) -> (B, n_patches, d_model)
        x = feed_forward(x)     // (B, n_patches, d_model) -> (B, n_patches, d_model)
    // Still (B, n_patches, d_model).

    // ----- Pool: one vector per sample -----
    IF pool_mode == "mean":
        x = mean(x, over dimension 1)   // (B, n_patches, d_model) -> (B, d_model)
    ELSE:
        x = x[:, last token, :]        // (B, d_model)
    // x is (B, d_model).

    // ----- Regression head -----
    out = Linear_head(x)   // (B, d_model) -> (B, 1)
    out = squeeze(out)     // (B, 1) -> (B,)
    RETURN out   // scalar per sample
```

---

## 6. Full training logic in pseudocode

```
FUNCTION train_patchtst(model, X_train, y_train, X_val, y_val, device, epochs, batch_size, lr, early_stopping_patience):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
    best_val_loss = infinity
    best_state = None
    patience_counter = 0
    history = { "train_loss": [], "val_loss": [] }

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    IF X_val and y_val provided and non-empty:
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    FOR epoch = 1 TO epochs:
        model.train()
        train_loss_sum = 0
        FOR each batch (xb, yb) in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)           // (batch,) scalars
            loss = criterion(pred, yb) // MSE
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        history["train_loss"].append(train_loss_sum / number_of_batches)

        IF val_loader exists:
            model.eval()
            val_loss_sum = 0
            FOR each batch (xb, yb) in val_loader (no gradients):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item()
            val_loss = val_loss_sum / number_of_val_batches
            history["val_loss"].append(val_loss)
            IF val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy of model.state_dict()
                patience_counter = 0
            ELSE:
                patience_counter += 1
            IF patience_counter >= early_stopping_patience:
                break
    IF best_state is not None:
        model.load_state_dict(best_state)
    RETURN history
```

**PatchTSTForecaster.fit** in addition: trim `seq_len` to a multiple of `patch_len`; fit `scaler_x` on flattened X and `scaler_y` on y; scale X and y; build the `PatchTST` net; call `train_patchtst`; store scalers and net. **Predict**: take last `seq_len` steps of X, scale with `scaler_x`, run `predict_patchtst`, inverse-transform with `scaler_y`.

---

## 7. Summary

- **Patching:** Split the time axis into non-overlapping chunks (e.g. 20 days → 5 patches of 4 days). Each chunk is one patch.
- **Patches → tokens:** Flatten each patch (patch_len × n_channels) and project with Linear to d_model; add optional positional embedding. The Transformer sees n_patches tokens.
- **Training:** Regression (MSE); scale X and y; train with optional val and early stopping; restore best weights.
- **Prediction:** Scale X, forward in batches, inverse-scale predictions.
- **Vs LSTM:** LSTM sequence = time steps; PatchTST sequence = patches (time chunks). Both use the time axis; PatchTST shortens it by patching.
- **Vs iTransformer:** iTransformer sequence = variables; PatchTST sequence = time patches. PatchTST keeps “time as the sequence” but in chunk form; iTransformer makes variables the sequence.

# Stacking Ensemble — Beginner’s Guide

This doc explains `src/ensemble/stacking.py`: what stacking is, why we use validation predictions, how `fit()`, `predict()`, and `evaluate()` work, the stacking logic in pseudocode, and when stacking helps vs when it fails.

---

## 1. What stacking is

### The idea in one sentence

**Stacking** means: you have several **base models** (e.g. LightGBM, LSTM, iTransformer, PatchTST). Instead of picking one, you train a **second model** (the **meta-model**) whose **inputs** are the **predictions** of the base models, and whose **output** is the final prediction. So the meta-model “learns how to combine” the base models.

### Simple picture

- **Level 0 (base models):** Each base model looks at the **raw features** (e.g. the 20-day window of 37 features) and outputs a prediction — here, **class probabilities** (e.g. P(down), P(up)).
- **Level 1 (meta-model):** It does **not** see the raw features. Its **input** is the list of base predictions for that sample. For example, for one sample:
  - LightGBM says (0.3, 0.7)  → prob down, prob up  
  - LSTM says (0.6, 0.4)  
  - iTransformer says (0.2, 0.8)  
  - So the meta-model’s input for this sample is the **concatenation**: [0.3, 0.7, 0.6, 0.4, 0.2, 0.8] (6 numbers for 3 models × 2 classes).  
  The meta-model then outputs the **final** class (or probabilities). So it learns things like “when LSTM and iTransformer both say up but LightGBM says down, prefer up” from data.

So **stacking = one model on top of others, trained to combine their outputs.**

### Why “stack”?

The base models are the first **layer** of prediction; the meta-model is the **second layer** stacked on top. So we “stack” two levels of models.

---

## 2. Why validation predictions are used

### The leakage problem

We want to train the meta-model on **base model outputs**. The question is: **which** base outputs?

- If we use the base models’ predictions on the **training set**: those predictions were made by models that were **trained on that same training set**. So the base models have already “seen” the training labels. Their training-set predictions can be **overconfident** or **overfit** to that set. If the meta-model learns from those, it learns from **contaminated** inputs — that’s **leakage**: information from the training labels flows into the meta-features in a too-direct way. The meta-model might do well on the training set but poorly on new data.

### The fix: use validation predictions

- We split data into **train**, **validation**, and **test** (in time order).
- Base models are trained **only on the training set** (and maybe use validation only for early stopping — they don’t use validation **labels** for training).
- We then run each base model on the **validation set** and collect their predictions. Those validation samples were **never used to train** the base models (we didn’t update parameters using val labels). So for the base models, validation predictions are **out-of-sample**: they’re “honest” predictions on unseen data.
- We train the **meta-model** on:
  - **Input:** those validation-set base predictions (stacked into one vector per sample).
  - **Target:** the **true validation labels** (y_val).

So the meta-model learns: “given these out-of-sample base predictions, what is the best way to combine them to get the true label?” No leakage: the meta-model’s inputs are base outputs that were not “taught” on those same samples.

**Summary:** We use **validation** predictions (not training predictions) so that the meta-model’s inputs are **out-of-sample** for the base models, avoiding leakage and giving a better signal for learning how to combine.

---

## 3. Walk through fit(), predict(), evaluate()

### fit(base_preds_val, y_val)

**What goes in:**

- **base_preds_val:** A dictionary. Each key is a model name (e.g. `"LightGBM"`, `"LSTM"`). Each value is an array of **predictions** for the **validation** set. Shape can be:
  - `(n_val,)` for binary (e.g. probability of class 1), or  
  - `(n_val, n_classes)` (probabilities for each class).
- **y_val:** The true labels for the validation set, shape `(n_val,)`, integers (0 or 1 for binary).

**What happens inside:**

1. **Stack:** For each validation sample, concatenate the probability vectors from all base models (in a fixed order, e.g. alphabetical by model name). So we get a matrix **X_meta** of shape `(n_val, n_models * n_classes)`. Example: 4 models, 2 classes → 8 numbers per sample.
2. **Train meta-model:** Fit a classifier (logistic regression or MLP) with:
   - **Input:** X_meta (stacked validation predictions).
   - **Target:** y_val (true validation labels).
3. Store the fitted meta-model and the list of model names (so we know the order of columns when we stack test predictions later).

**What comes out:** The same ensemble object (`self`) is returned, now fitted. You can call `predict` or `evaluate` next.

**Why:** This is the whole “training” of the ensemble: the meta-model learns how to map base predictions → final class from validation data.

---

### predict(base_preds_test)

**What goes in:**

- **base_preds_test:** Same structure as for fit, but for the **test** set: `{model_name: array of shape (n_test,) or (n_test, n_classes)}`.

**What happens inside:**

1. **Stack:** Same as in fit: concatenate base predictions per sample → matrix of shape `(n_test, n_models * n_classes)`.
2. **Meta-model predict:** Run the fitted meta-model on this matrix. It outputs class probabilities, shape `(n_test, n_classes)`.
3. **predict()** returns the **class** (argmax of those probabilities), i.e. shape `(n_test,)` integers.

**What comes out:** Array of predicted labels (0 or 1 for binary), length n_test.

**Why:** At test time we don’t have labels; we only have base predictions. The meta-model turns those into the final decision.

---

### predict_proba(base_preds_test)

Same as `predict`, but the return value is the **probability** matrix from the meta-model, shape `(n_test, n_classes)`, instead of the argmax. Used when we need probabilities (e.g. for ROC-AUC).

---

### evaluate(y_test, base_preds_test)

**What goes in:**

- **y_test:** True labels for the test set, shape `(n_test,)`.
- **base_preds_test:** Base model predictions on the test set (same as for `predict`).

**What happens inside:**

1. Get ensemble predictions: `y_pred = self.predict(base_preds_test)` and `y_prob = self.predict_proba(base_preds_test)`.
2. Call the **same** evaluation function used for single models (`evaluate_trend`): it takes (y_test, y_pred, y_prob) and returns accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

**What comes out:** A dictionary of metrics (accuracy, precision, recall, f1, roc_auc, confusion_matrix).

**Why:** So we can compare the **ensemble** with each base model using the same metrics on the same test set.

---

## 4. Stacking logic in pseudocode

```
// ----- Setup (before stacking) -----
// You already have:
//   - Base models trained on (X_train, y_train), optionally using (X_val, y_val) for early stopping only.
//   - base_preds_val = { "LightGBM": (n_val, 2), "LSTM": (n_val, 2), ... }  // probabilities on VAL
//   - y_val = (n_val,) true labels
//   - base_preds_test = { "LightGBM": (n_test, 2), "LSTM": (n_test, 2), ... }  // probabilities on TEST
//   - y_test = (n_test,) true labels

FUNCTION stack_predictions(base_preds, n_classes):
    names = sorted keys of base_preds
    FOR each name in names:
        P = base_preds[name]
        Ensure P has shape (n_samples, n_classes) [convert 1d to 2d if binary]
        append P to list parts
    X_meta = concatenate parts along columns   // (n_samples, n_models * n_classes)
    RETURN X_meta

// ----- Fit (train the meta-model) -----
FUNCTION fit(base_preds_val, y_val):
    X_meta = stack_predictions(base_preds_val, n_classes)   // (n_val, n_models * n_classes)
    meta_model = LogisticRegression(...)   // or MLP
    meta_model.fit(X_meta, y_val)
    Store meta_model and list of model names
    RETURN self

// ----- Predict -----
FUNCTION predict(base_preds_test):
    X_meta = stack_predictions(base_preds_test, n_classes)   // (n_test, n_models * n_classes)
    probs = meta_model.predict_proba(X_meta)   // (n_test, n_classes)
    RETURN argmax(probs, axis=1)   // (n_test,) integer labels

FUNCTION predict_proba(base_preds_test):
    X_meta = stack_predictions(base_preds_test, n_classes)
    RETURN meta_model.predict_proba(X_meta)   // (n_test, n_classes)

// ----- Evaluate -----
FUNCTION evaluate(y_test, base_preds_test):
    y_pred = predict(base_preds_test)
    y_prob = predict_proba(base_preds_test)
    RETURN evaluate_trend(y_test, y_pred, y_prob, n_classes)   // accuracy, f1, roc_auc, etc.
```

So the core is: **stack base predictions into X_meta → meta-model fits on (X_meta_val, y_val) → at test time stack base test predictions → meta-model predicts from that.**

---

## 5. When stacking helps and when it fails

### When stacking usually helps

1. **Base models make different kinds of mistakes.**  
   If one model is wrong when another is right, the meta-model can learn “trust LSTM more when LightGBM is uncertain” or “when both say up, say up.” So **diversity** among base models (different architectures: tree, LSTM, Transformer, etc.) often helps.

2. **Base models are at least somewhat good.**  
   If every base model is only 50% accurate (random), the meta-model has no signal to learn from. Stacking improves when base models are **individually useful** but not perfect.

3. **Enough validation data.**  
   The meta-model needs enough (X_meta_val, y_val) pairs to learn a stable combination. If validation is tiny, the meta-model can overfit to val and not generalize to test.

4. **Base predictions are well aligned.**  
   In this code, all base models output predictions for the **same** validation (and test) samples — same rows, same order. So the stacked vector for sample i really is “what every model said for sample i.” If alignment is wrong (e.g. different sample sets or order), stacking breaks.

### When stacking can fail or add little

1. **Base models are too similar.**  
   If all base models are almost the same (e.g. same architecture, same data), their predictions are highly correlated. Then the meta-model has almost no extra information — it’s like having one input repeated. Little or no gain.

2. **One model is much better than the others.**  
   If LightGBM is 80% accurate and the rest are 52%, the best strategy might be “always follow LightGBM.” The meta-model might learn that (which is fine), but you could have just used LightGBM alone. Stacking helps most when **no single model dominates** and combining adds value.

3. **Validation set is too small or unrepresentative.**  
   If val is small, the meta-model may overfit to val. If val has a different distribution than test (e.g. different time period in finance), the learned combination may not transfer. So stacking can **fail** when val is not representative of test.

4. **Meta-model is too complex.**  
   If you use a very flexible meta-model (e.g. a big MLP) on a small validation set, it can overfit and the ensemble can do **worse** than the best base model. Simple meta-models (e.g. logistic regression) are often more robust.

5. **Base models are all bad.**  
   If every base model is poor, stacking cannot magically fix that. “Garbage in, garbage out” — the meta-model only combines what it’s given.

### Rule of thumb

- **Try stacking** when you have several **different** base models that are **reasonably good** and a **sufficient, representative** validation set.  
- **Don’t expect a miracle** when base models are very similar or when one model is already much better than the rest.  
- **Keep the meta-model simple** (e.g. logistic regression) unless you have a lot of validation data and a reason to use something more complex.

In this project, stacking is used to combine LightGBM, LSTM, iTransformer, and (optionally) PatchTST — different families of models — so there is enough diversity for the meta-model to learn a useful combination when conditions above are met.

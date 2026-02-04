# Tail-Aware Training
## Tập trung vào Extreme Events trong Financial ML

---

## Mục lục

1. [Tại sao cần Tail-Aware Training?](#1-tại-sao-cần-tail-aware-training)
2. [Công thức Tail-Aware Loss](#2-công-thức-tail-aware-loss)
3. [Threshold Selection](#3-threshold-selection)
4. [Implementation](#4-implementation)
5. [So sánh với Standard Loss](#5-so-sánh-với-standard-loss)
6. [Best Practices](#6-best-practices)
7. [Bài tập thực hành](#7-bài-tập-thực-hành)

---

## 1. TẠI SAO CẦN TAIL-AWARE TRAINING?

### 1.1. Vấn đề với Standard Loss

**Standard MSE/MAE:**
```
Loss = (1/n) × Σ Error(i)

Mọi samples đều bình đẳng:
- Error 1% trên ngày bình thường
- Error 1% trên ngày crash
→ Đóng góp BẰNG NHAU vào loss!
```

**Hậu quả:**
```
Distribution của returns:
     
     │        ▄█▄
     │       ▄███▄
     │      ▄█████▄
     │     ▄███████▄
     │    ▄█████████▄
     │   ▄███████████▄
     │▄▄▄█████████████▄▄▄
     └────────────────────→ return
     -10%        0%        +10%
        ↑                    ↑
      TAILS              TAILS
    (5% samples)       (5% samples)

Standard Loss:
- 90% samples ở center → dominate loss
- 5% extreme negatives → bị ignore
- 5% extreme positives → bị ignore

Model optimize cho AVERAGE case
→ FAIL trên extreme events!
```

### 1.2. Tail Events trong Finance

**Tại sao Tails quan trọng:**
```
Average day:     Return = +0.05%, Risk = nhỏ
Tail day (5%):   Return = -5%, Risk = HUGE

Một ngày -10% có thể:
- Wipe out profits của 100 ngày +0.1%
- Trigger margin calls
- Break risk limits

→ Dự đoán đúng TAILS quan trọng hơn average!
```

**Ví dụ thực tế:**
```
FPT 2020:
- 245/252 ngày: Return trong [-2%, +2%]
- 7 ngày: Return ngoài [-5%, +5%]

Model A (standard): 
  MAE overall = 1.5%
  MAE on tails = 8%

Model B (tail-aware):
  MAE overall = 1.8%
  MAE on tails = 4%

→ Model B TỆ HƠN về overall nhưng TỐT HƠN cho risk!
```

---

## 2. CÔNG THỨC TAIL-AWARE LOSS

### 2.1. Tail MSE (Top-k% Loss)

**Ý tưởng:** Chỉ tính loss trên k% samples có error lớn nhất.

**Công thức:**
```
Tail_MSE_k = (1/m) × Σ Error(i)²    cho i ∈ Top-k% |Error|

Trong đó:
- m = k% × n = số samples trong tail
- Top-k% |Error| = samples có |Error| lớn nhất
```

**Implementation:**
```python
import numpy as np
import torch
import torch.nn as nn

def tail_mse_loss(y_true, y_pred, k_percent=0.1):
    """
    Tail MSE: Chỉ tính loss trên top-k% errors
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        k_percent: Percentage of worst errors to consider (0.1 = 10%)
    
    Returns:
        Tail MSE
    
    Ví dụ:
    - 1000 samples, k=10%
    - Tính MSE chỉ trên 100 samples có error lớn nhất
    """
    errors = (y_true - y_pred) ** 2
    
    # Số samples trong tail
    n_tail = max(1, int(len(errors) * k_percent))
    
    # Sort errors descending, lấy top k%
    sorted_errors = np.sort(errors)[::-1]
    tail_errors = sorted_errors[:n_tail]
    
    return np.mean(tail_errors)


class TailMSELoss(nn.Module):
    """Tail MSE Loss cho PyTorch"""
    
    def __init__(self, k_percent=0.1):
        super().__init__()
        self.k_percent = k_percent
    
    def forward(self, y_pred, y_true):
        errors = (y_pred - y_true) ** 2
        
        n_tail = max(1, int(len(errors) * self.k_percent))
        
        # TopK trong PyTorch
        tail_errors, _ = torch.topk(errors.squeeze(), n_tail)
        
        return torch.mean(tail_errors)
```

### 2.2. Tail MAE (Top-k% Absolute Loss)

**Công thức:**
```
Tail_MAE_k = (1/m) × Σ |Error(i)|    cho i ∈ Top-k% |Error|
```

```python
def tail_mae_loss(y_true, y_pred, k_percent=0.1):
    """Tail MAE: Top-k% absolute errors"""
    errors = np.abs(y_true - y_pred)
    n_tail = max(1, int(len(errors) * k_percent))
    
    sorted_errors = np.sort(errors)[::-1]
    tail_errors = sorted_errors[:n_tail]
    
    return np.mean(tail_errors)


class TailMAELoss(nn.Module):
    """Tail MAE Loss cho PyTorch"""
    
    def __init__(self, k_percent=0.1):
        super().__init__()
        self.k_percent = k_percent
    
    def forward(self, y_pred, y_true):
        errors = torch.abs(y_pred - y_true)
        n_tail = max(1, int(len(errors) * self.k_percent))
        
        tail_errors, _ = torch.topk(errors.squeeze(), n_tail)
        return torch.mean(tail_errors)
```

### 2.3. Quantile-Based Tail Loss

**Ý tưởng:** Dùng quantile thay vì top-k%.

**Công thức:**
```
Tail_Loss_q = Mean(Error(i))    cho i mà |Error(i)| > Quantile_q(|Error|)

Ví dụ q=0.9:
- Tính quantile 90% của |Error|
- Chỉ tính loss trên errors > quantile này
```

```python
def quantile_tail_loss(y_true, y_pred, quantile=0.9, loss_type='mse'):
    """
    Quantile-based Tail Loss
    
    Args:
        quantile: Threshold quantile (0.9 = top 10%)
        loss_type: 'mse' hoặc 'mae'
    """
    errors = np.abs(y_true - y_pred)
    threshold = np.quantile(errors, quantile)
    
    # Mask cho tail samples
    tail_mask = errors > threshold
    
    if loss_type == 'mse':
        tail_errors = (y_true[tail_mask] - y_pred[tail_mask]) ** 2
    else:  # mae
        tail_errors = errors[tail_mask]
    
    return np.mean(tail_errors)


class QuantileTailLoss(nn.Module):
    """Quantile-based Tail Loss cho PyTorch"""
    
    def __init__(self, quantile=0.9, loss_type='mse'):
        super().__init__()
        self.quantile = quantile
        self.loss_type = loss_type
    
    def forward(self, y_pred, y_true):
        errors = torch.abs(y_pred - y_true)
        
        # Compute quantile
        threshold = torch.quantile(errors, self.quantile)
        
        # Mask
        tail_mask = errors > threshold
        
        if self.loss_type == 'mse':
            tail_errors = (y_pred[tail_mask] - y_true[tail_mask]) ** 2
        else:
            tail_errors = errors[tail_mask]
        
        # Handle empty tail
        if len(tail_errors) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return torch.mean(tail_errors)
```

### 2.4. Return-Based Tail Loss

**Ý tưởng:** Focus vào samples có |return| lớn (extreme market movements).

**Khác biệt:**
```
Error-based tail: Samples có prediction error lớn nhất
Return-based tail: Samples có actual return lớn nhất (extreme days)

Return-based phù hợp hơn cho financial:
- Ta BIẾT TRƯỚC ngày nào là extreme (based on return)
- Ta muốn model predict tốt trên những ngày đó
```

```python
def return_based_tail_loss(y_true, y_pred, returns, quantile=0.95):
    """
    Loss trên extreme return days
    
    Args:
        returns: Actual returns (để xác định extreme days)
        quantile: 0.95 = top 5% |return| days
    """
    abs_returns = np.abs(returns)
    threshold = np.quantile(abs_returns, quantile)
    
    # Extreme days mask
    extreme_mask = abs_returns > threshold
    
    # Loss chỉ trên extreme days
    extreme_errors = (y_true[extreme_mask] - y_pred[extreme_mask]) ** 2
    
    return np.mean(extreme_errors)


class ReturnBasedTailLoss(nn.Module):
    """
    Loss tập trung vào extreme return days
    
    Cần pass returns như additional input
    """
    
    def __init__(self, quantile=0.95):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, y_pred, y_true, returns):
        abs_returns = torch.abs(returns)
        threshold = torch.quantile(abs_returns, self.quantile)
        
        extreme_mask = abs_returns > threshold
        
        if extreme_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        extreme_errors = (y_pred[extreme_mask] - y_true[extreme_mask]) ** 2
        return torch.mean(extreme_errors)
```

### 2.5. Combined Loss (Standard + Tail)

**Ý tưởng:** Optimize cho cả average và extreme cases.

**Công thức:**
```
Combined_Loss = (1 - λ) × Standard_Loss + λ × Tail_Loss

Trong đó λ ∈ [0, 1] là trade-off parameter
- λ = 0: Pure standard loss
- λ = 1: Pure tail loss
- λ = 0.3: 70% standard, 30% tail (recommended start)
```

```python
class CombinedTailLoss(nn.Module):
    """
    Combined Loss = Standard + Tail
    
    Best of both worlds:
    - Không hoàn toàn ignore average cases
    - Nhưng emphasize extreme cases
    """
    
    def __init__(self, tail_weight=0.3, tail_quantile=0.9, loss_type='mse'):
        super().__init__()
        self.tail_weight = tail_weight
        self.tail_quantile = tail_quantile
        self.loss_type = loss_type
    
    def forward(self, y_pred, y_true):
        # Standard loss (all samples)
        if self.loss_type == 'mse':
            standard_loss = torch.mean((y_pred - y_true) ** 2)
        else:
            standard_loss = torch.mean(torch.abs(y_pred - y_true))
        
        # Tail loss
        errors = torch.abs(y_pred - y_true)
        threshold = torch.quantile(errors, self.tail_quantile)
        tail_mask = errors > threshold
        
        if tail_mask.sum() > 0:
            if self.loss_type == 'mse':
                tail_loss = torch.mean((y_pred[tail_mask] - y_true[tail_mask]) ** 2)
            else:
                tail_loss = torch.mean(errors[tail_mask])
        else:
            tail_loss = standard_loss
        
        # Combined
        combined = (1 - self.tail_weight) * standard_loss + self.tail_weight * tail_loss
        
        return combined


# Usage
criterion = CombinedTailLoss(
    tail_weight=0.3,      # 30% weight cho tail loss
    tail_quantile=0.9,    # Top 10% errors
    loss_type='mse'
)
```

---

## 3. THRESHOLD SELECTION

### 3.1. Chọn k% cho Top-k

**Guidelines:**
```
k = 5%:  Rất extreme, có thể unstable (ít samples)
k = 10%: Recommended cho financial
k = 20%: Moderate tail focus
k = 30%: Mild tail focus
```

**Cách chọn dựa trên data:**
```python
def analyze_tail_distribution(returns, k_values=[0.05, 0.1, 0.15, 0.2]):
    """
    Phân tích tail distribution để chọn k
    """
    print("=== TAIL DISTRIBUTION ANALYSIS ===\n")
    
    for k in k_values:
        threshold = np.quantile(np.abs(returns), 1 - k)
        tail_mask = np.abs(returns) > threshold
        
        n_tail = tail_mask.sum()
        mean_abs_return = np.abs(returns[tail_mask]).mean()
        max_abs_return = np.abs(returns[tail_mask]).max()
        
        print(f"k = {k*100:.0f}%:")
        print(f"  Threshold: |return| > {threshold*100:.2f}%")
        print(f"  N samples: {n_tail}")
        print(f"  Mean |return|: {mean_abs_return*100:.2f}%")
        print(f"  Max |return|: {max_abs_return*100:.2f}%")
        print()

# Usage
analyze_tail_distribution(df['return'].values)
```

### 3.2. Adaptive Threshold

```python
class AdaptiveTailLoss(nn.Module):
    """
    Threshold adapt theo volatility của window gần nhất
    
    High vol period: Threshold cao hơn
    Low vol period: Threshold thấp hơn
    """
    
    def __init__(self, base_quantile=0.9, vol_window=20):
        super().__init__()
        self.base_quantile = base_quantile
        self.vol_window = vol_window
    
    def forward(self, y_pred, y_true, recent_volatility):
        """
        Args:
            recent_volatility: Rolling volatility để adjust threshold
        """
        # Adjust quantile based on recent vol
        # High vol → higher quantile (fewer extreme samples)
        vol_ratio = recent_volatility / recent_volatility.mean()
        adjusted_quantile = self.base_quantile * vol_ratio
        adjusted_quantile = torch.clamp(adjusted_quantile, 0.8, 0.99)
        
        # Compute loss với adjusted threshold
        errors = torch.abs(y_pred - y_true)
        losses = []
        
        for i, q in enumerate(adjusted_quantile):
            if errors[:i+1].numel() > 10:  # Need enough history
                threshold = torch.quantile(errors[:i+1], q)
                if errors[i] > threshold:
                    losses.append((y_pred[i] - y_true[i]) ** 2)
        
        if len(losses) == 0:
            return torch.mean((y_pred - y_true) ** 2)
        
        return torch.mean(torch.stack(losses))
```

---

## 4. IMPLEMENTATION

### 4.1. Full Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TailAwareTrainer:
    """
    Trainer với Tail-Aware Loss
    """
    
    def __init__(self, model, loss_config):
        """
        Args:
            loss_config: dict với:
                - type: 'tail_mse', 'tail_mae', 'combined', 'quantile'
                - k_percent: cho top-k loss
                - quantile: cho quantile loss
                - tail_weight: cho combined loss
        """
        self.model = model
        self.loss_config = loss_config
        self.criterion = self._create_loss()
    
    def _create_loss(self):
        loss_type = self.loss_config.get('type', 'combined')
        
        if loss_type == 'tail_mse':
            return TailMSELoss(k_percent=self.loss_config.get('k_percent', 0.1))
        elif loss_type == 'tail_mae':
            return TailMAELoss(k_percent=self.loss_config.get('k_percent', 0.1))
        elif loss_type == 'quantile':
            return QuantileTailLoss(
                quantile=self.loss_config.get('quantile', 0.9)
            )
        elif loss_type == 'combined':
            return CombinedTailLoss(
                tail_weight=self.loss_config.get('tail_weight', 0.3),
                tail_quantile=self.loss_config.get('quantile', 0.9)
            )
        else:
            return nn.MSELoss()
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for X, y in dataloader:
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.FloatTensor(X))
            
            # Standard metrics
            mse = torch.mean((y_pred - torch.FloatTensor(y).unsqueeze(1)) ** 2).item()
            mae = torch.mean(torch.abs(y_pred - torch.FloatTensor(y).unsqueeze(1))).item()
            
            # Tail metrics
            errors = torch.abs(y_pred.squeeze() - torch.FloatTensor(y))
            k = int(len(errors) * 0.1)
            tail_errors, _ = torch.topk(errors, k)
            tail_mae = torch.mean(tail_errors).item()
        
        return {
            'MSE': mse,
            'MAE': mae,
            'Tail_MAE_10': tail_mae,
            'RMSE': np.sqrt(mse)
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, lr=0.001, batch_size=32, early_stopping=10):
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience = 0
        history = {'train_loss': [], 'val_metrics': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            
            if X_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                history['val_metrics'].append(val_metrics)
                
                # Early stopping on tail metric
                val_tail_loss = val_metrics['Tail_MAE_10']
                if val_tail_loss < best_val_loss:
                    best_val_loss = val_tail_loss
                    patience = 0
                    torch.save(self.model.state_dict(), 'best_tail_model.pt')
                else:
                    patience += 1
                    if patience >= early_stopping:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}", end='')
                if X_val is not None:
                    print(f", Val MAE = {val_metrics['MAE']:.4f}, Val Tail MAE = {val_metrics['Tail_MAE_10']:.4f}")
                else:
                    print()
        
        # Load best model
        if X_val is not None:
            self.model.load_state_dict(torch.load('best_tail_model.pt'))
        
        return history


# Usage
model = YourModel(input_dim=X_train.shape[1])

loss_config = {
    'type': 'combined',
    'tail_weight': 0.3,
    'quantile': 0.9
}

trainer = TailAwareTrainer(model, loss_config)
history = trainer.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    lr=0.001,
    early_stopping=15
)
```

### 4.2. sklearn-Compatible Wrapper

```python
from sklearn.base import BaseEstimator, RegressorMixin

class TailAwareRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible wrapper cho Tail-Aware models
    """
    
    def __init__(self, 
                 hidden_dim=64,
                 tail_weight=0.3,
                 tail_quantile=0.9,
                 epochs=100,
                 lr=0.001,
                 batch_size=32):
        self.hidden_dim = hidden_dim
        self.tail_weight = tail_weight
        self.tail_quantile = tail_quantile
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
    
    def fit(self, X, y):
        self.input_dim = X.shape[1]
        
        # Create model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Loss
        criterion = CombinedTailLoss(
            tail_weight=self.tail_weight,
            tail_quantile=self.tail_quantile
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.FloatTensor(X))
        return y_pred.numpy().squeeze()


# Usage với sklearn
from sklearn.model_selection import cross_val_score

model = TailAwareRegressor(
    hidden_dim=64,
    tail_weight=0.3,
    epochs=50
)

# Cross-validation
# Note: Nên dùng TimeSeriesSplit thay vì standard CV
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
print(f"CV MAE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 5. SO SÁNH VỚI STANDARD LOSS

### 5.1. Experiment Setup

```python
def compare_loss_functions(X_train, y_train, X_test, y_test, 
                          returns_test=None):
    """
    So sánh Standard vs Tail-Aware Loss
    """
    from sklearn.neural_network import MLPRegressor
    
    results = []
    
    # Model 1: Standard MSE
    model_standard = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
    model_standard.fit(X_train, y_train)
    pred_standard = model_standard.predict(X_test)
    
    # Model 2: Tail-Aware (dùng custom wrapper)
    model_tail = TailAwareRegressor(
        hidden_dim=64,
        tail_weight=0.3,
        epochs=200
    )
    model_tail.fit(X_train, y_train)
    pred_tail = model_tail.predict(X_test)
    
    # Evaluate on ALL samples
    mae_all_std = np.mean(np.abs(y_test - pred_standard))
    mae_all_tail = np.mean(np.abs(y_test - pred_tail))
    
    # Evaluate on TAIL samples (top 10% |return|)
    if returns_test is not None:
        threshold = np.quantile(np.abs(returns_test), 0.9)
        tail_mask = np.abs(returns_test) > threshold
        
        mae_tail_std = np.mean(np.abs(y_test[tail_mask] - pred_standard[tail_mask]))
        mae_tail_tail = np.mean(np.abs(y_test[tail_mask] - pred_tail[tail_mask]))
    else:
        # Use prediction errors instead
        errors_std = np.abs(y_test - pred_standard)
        threshold = np.quantile(errors_std, 0.9)
        tail_mask = errors_std > threshold
        
        mae_tail_std = np.mean(errors_std[tail_mask])
        mae_tail_tail = np.mean(np.abs(y_test[tail_mask] - pred_tail[tail_mask]))
    
    print("=== COMPARISON: Standard vs Tail-Aware Loss ===")
    print()
    print("Overall Performance (All Samples):")
    print(f"  Standard MSE:   MAE = {mae_all_std:.4f}")
    print(f"  Tail-Aware:     MAE = {mae_all_tail:.4f}")
    print(f"  Difference:     {(mae_all_tail - mae_all_std) / mae_all_std * 100:+.1f}%")
    print()
    print("Tail Performance (Top 10% Extreme Days):")
    print(f"  Standard MSE:   MAE = {mae_tail_std:.4f}")
    print(f"  Tail-Aware:     MAE = {mae_tail_tail:.4f}")
    print(f"  Improvement:    {(mae_tail_std - mae_tail_tail) / mae_tail_std * 100:+.1f}%")
    
    return {
        'mae_all_standard': mae_all_std,
        'mae_all_tail': mae_all_tail,
        'mae_extreme_standard': mae_tail_std,
        'mae_extreme_tail': mae_tail_tail
    }
```

### 5.2. Trade-off Analysis

```python
def analyze_tradeoff(X_train, y_train, X_test, y_test,
                    tail_weights=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]):
    """
    Phân tích trade-off giữa overall và tail performance
    """
    results = []
    
    for tw in tail_weights:
        model = TailAwareRegressor(
            hidden_dim=64,
            tail_weight=tw,
            epochs=100
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # Overall MAE
        mae_overall = np.mean(np.abs(y_test - pred))
        
        # Tail MAE (top 10%)
        errors = np.abs(y_test - pred)
        threshold = np.quantile(errors, 0.9)
        mae_tail = np.mean(errors[errors > threshold])
        
        results.append({
            'tail_weight': tw,
            'mae_overall': mae_overall,
            'mae_tail': mae_tail
        })
    
    df_results = pd.DataFrame(results)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_results['tail_weight'], df_results['mae_overall'], 
            'b-o', label='Overall MAE')
    ax.plot(df_results['tail_weight'], df_results['mae_tail'], 
            'r-o', label='Tail MAE')
    
    ax.set_xlabel('Tail Weight (λ)')
    ax.set_ylabel('MAE')
    ax.set_title('Trade-off: Overall vs Tail Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    return df_results
```

---

## 6. BEST PRACTICES

### 6.1. Khi nào dùng Tail-Aware Loss

**Nên dùng:**
```
✓ Risk management là priority
✓ Extreme events có impact lớn (finance, safety)
✓ Có đủ samples trong tail (>30 samples)
✓ Tail có pattern học được
```

**Không nên dùng:**
```
✗ Tất cả samples equally important
✗ Tail quá ít samples (unstable)
✗ Tail là pure noise (không có pattern)
```

### 6.2. Hyperparameter Selection

```python
def select_tail_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Grid search cho tail loss hyperparameters
    """
    param_grid = {
        'tail_weight': [0.1, 0.2, 0.3, 0.5],
        'tail_quantile': [0.85, 0.9, 0.95]
    }
    
    best_score = float('inf')
    best_params = None
    
    for tw in param_grid['tail_weight']:
        for tq in param_grid['tail_quantile']:
            model = TailAwareRegressor(
                tail_weight=tw,
                tail_quantile=tq,
                epochs=50  # Quick training for search
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            # Score: combination of overall and tail MAE
            errors = np.abs(y_val - pred)
            mae_overall = np.mean(errors)
            
            threshold = np.quantile(errors, tq)
            mae_tail = np.mean(errors[errors > threshold])
            
            # Weighted score (emphasize tail)
            score = (1 - tw) * mae_overall + tw * mae_tail
            
            print(f"tw={tw}, tq={tq}: Overall MAE={mae_overall:.4f}, Tail MAE={mae_tail:.4f}, Score={score:.4f}")
            
            if score < best_score:
                best_score = score
                best_params = {'tail_weight': tw, 'tail_quantile': tq}
    
    print(f"\nBest params: {best_params}")
    return best_params
```

### 6.3. Monitoring và Debugging

```python
class TailLossCallback:
    """Callback để monitor tail loss during training"""
    
    def __init__(self, X_val, y_val, log_interval=10):
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        self.log_interval = log_interval
        self.history = []
    
    def on_epoch_end(self, epoch, model):
        if epoch % self.log_interval != 0:
            return
        
        model.eval()
        with torch.no_grad():
            pred = model(self.X_val).squeeze()
            errors = torch.abs(pred - self.y_val)
            
            mae_overall = torch.mean(errors).item()
            
            # Tail metrics at different thresholds
            for q in [0.9, 0.95]:
                threshold = torch.quantile(errors, q)
                tail_mask = errors > threshold
                mae_tail = torch.mean(errors[tail_mask]).item()
                
                self.history.append({
                    'epoch': epoch,
                    'mae_overall': mae_overall,
                    f'mae_tail_{int(q*100)}': mae_tail
                })
        
        print(f"Epoch {epoch}: Overall MAE={mae_overall:.4f}, Tail90 MAE={mae_tail:.4f}")
```

---

## 7. BÀI TẬP THỰC HÀNH

### Bài tập 1: Implement Tail Losses

**Yêu cầu:**
1. Implement TailMSELoss và TailMAELoss
2. Test trên synthetic data
3. Verify gradient flow

### Bài tập 2: Compare Loss Functions

**Yêu cầu:**
1. Train 3 models: Standard, Tail-only, Combined
2. Evaluate trên overall và tail metrics
3. Visualize trade-off

### Bài tập 3: Hyperparameter Tuning

**Yêu cầu:**
1. Grid search cho tail_weight và quantile
2. Use time series CV
3. Report optimal parameters

### Bài tập 4: Real-world Application

**Yêu cầu:**
1. Apply tail-aware training cho VN30 prediction
2. Backtest với risk-adjusted metrics
3. Compare Sharpe ratio

---

## Kiểm tra hiểu bài

- [ ] Giải thích được tại sao standard loss không tốt cho tails
- [ ] Implement được Tail MSE và Tail MAE
- [ ] Chọn được threshold/quantile phù hợp
- [ ] Analyze được trade-off overall vs tail performance
- [ ] Integrate được vào training loop

---

## Tài liệu tham khảo

**Papers:**
- "Deep Learning for Tail Risk" - various authors
- "Extreme Value Theory in Finance" - McNeil, Frey, Embrechts

**Related:**
- Focal Loss (object detection - similar idea)
- Quantile Regression
- CVaR Optimization

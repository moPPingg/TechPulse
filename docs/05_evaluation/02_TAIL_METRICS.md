# Tail-Aware Evaluation Metrics
## Đánh giá Model trên Extreme Events

---

## Mục lục

1. [Tại sao cần Tail Metrics?](#1-tại-sao-cần-tail-metrics)
2. [Tail-MAE và Tail-RMSE](#2-tail-mae-và-tail-rmse)
3. [Event-Window Metrics](#3-event-window-metrics)
4. [Directional Accuracy trong Tail](#4-directional-accuracy-trong-tail)
5. [Shock Recall và Precision](#5-shock-recall-và-precision)
6. [Comprehensive Evaluation Framework](#6-comprehensive-evaluation-framework)
7. [Implementation](#7-implementation)
8. [Bài tập thực hành](#8-bài-tập-thực-hành)

---

## 1. TẠI SAO CẦN TAIL METRICS?

### 1.1. Vấn đề với Standard Metrics

**Standard MAE/RMSE:**
```
Model A: MAE = 1.5% (looks good!)
Model B: MAE = 1.8% (looks worse)

Nhưng:
Model A:
  - Normal days: MAE = 1.2%
  - Extreme days: MAE = 8.0%  ← TERRIBLE!

Model B:
  - Normal days: MAE = 1.5%
  - Extreme days: MAE = 3.5%  ← MUCH BETTER!

Standard metrics KHÔNG cho thấy điều này!
```

### 1.2. Extreme Days Matter More

```
Trong trading:
- 95% ngày: Small returns [-2%, +2%]
- 5% ngày: Large returns (±5% hoặc hơn)

Impact:
- 100 ngày average: Profit $100
- 1 ngày extreme (-10%): Loss $500

→ Dự đoán đúng 1 ngày extreme quan trọng hơn 100 ngày average!
```

### 1.3. Tail Metrics giải quyết gì

```
Tail Metrics cho biết:
1. Model performs như thế nào trên extreme days
2. Model có detect được direction của extreme moves không
3. Model có "thấy trước" shocks không
4. Trade-off giữa overall và tail performance

→ Đánh giá REALISTIC cho trading/risk management
```

---

## 2. TAIL-MAE VÀ TAIL-RMSE

### 2.1. Định nghĩa

**Tail-MAE (Top-k%):**
```
Tail_MAE_k = (1/m) × Σ|y_true(i) - y_pred(i)|
             cho i ∈ Top k% |actual return|

m = số samples trong tail (k% × n)
```

**Tail-RMSE (Top-k%):**
```
Tail_RMSE_k = √[(1/m) × Σ(y_true(i) - y_pred(i))²]
              cho i ∈ Top k% |actual return|
```

### 2.2. Implementation

```python
import numpy as np
from typing import Dict, List, Optional

def tail_mae(y_true: np.ndarray, 
             y_pred: np.ndarray, 
             returns: np.ndarray,
             k_percent: float = 0.1) -> float:
    """
    Tail MAE: MAE chỉ trên top k% |return| days
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values
        returns: Actual returns (để xác định extreme days)
        k_percent: Percentage of extreme days (0.1 = 10%)
    
    Returns:
        Tail MAE
    """
    abs_returns = np.abs(returns)
    threshold = np.quantile(abs_returns, 1 - k_percent)
    
    # Mask cho extreme days
    tail_mask = abs_returns >= threshold
    
    if tail_mask.sum() == 0:
        return np.nan
    
    errors = np.abs(y_true[tail_mask] - y_pred[tail_mask])
    return np.mean(errors)


def tail_rmse(y_true: np.ndarray,
              y_pred: np.ndarray,
              returns: np.ndarray,
              k_percent: float = 0.1) -> float:
    """
    Tail RMSE: RMSE chỉ trên top k% |return| days
    """
    abs_returns = np.abs(returns)
    threshold = np.quantile(abs_returns, 1 - k_percent)
    
    tail_mask = abs_returns >= threshold
    
    if tail_mask.sum() == 0:
        return np.nan
    
    errors = (y_true[tail_mask] - y_pred[tail_mask]) ** 2
    return np.sqrt(np.mean(errors))


def tail_mape(y_true: np.ndarray,
              y_pred: np.ndarray,
              returns: np.ndarray,
              k_percent: float = 0.1) -> float:
    """
    Tail MAPE: MAPE chỉ trên top k% |return| days
    
    Note: Cẩn thận với y_true gần 0
    """
    abs_returns = np.abs(returns)
    threshold = np.quantile(abs_returns, 1 - k_percent)
    
    tail_mask = abs_returns >= threshold
    
    if tail_mask.sum() == 0:
        return np.nan
    
    # Avoid division by zero
    y_true_tail = y_true[tail_mask]
    y_pred_tail = y_pred[tail_mask]
    
    valid_mask = np.abs(y_true_tail) > 1e-8
    if valid_mask.sum() == 0:
        return np.nan
    
    errors = np.abs((y_true_tail[valid_mask] - y_pred_tail[valid_mask]) / y_true_tail[valid_mask])
    return np.mean(errors) * 100
```

### 2.3. Multiple Tail Levels

```python
def multi_level_tail_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             returns: np.ndarray,
                             k_levels: List[float] = [0.05, 0.1, 0.2]) -> Dict:
    """
    Tính tail metrics ở nhiều levels
    
    Args:
        k_levels: List các percentages [0.05, 0.1, 0.2] = [5%, 10%, 20%]
    
    Returns:
        Dict với metrics cho mỗi level
    """
    results = {}
    
    for k in k_levels:
        level_name = f"tail_{int(k*100)}pct"
        
        results[f"{level_name}_mae"] = tail_mae(y_true, y_pred, returns, k)
        results[f"{level_name}_rmse"] = tail_rmse(y_true, y_pred, returns, k)
        
        # Số samples trong tail
        abs_returns = np.abs(returns)
        threshold = np.quantile(abs_returns, 1 - k)
        results[f"{level_name}_n_samples"] = (abs_returns >= threshold).sum()
    
    return results

# Usage
metrics = multi_level_tail_metrics(y_test, y_pred, returns_test)
print("Tail Metrics at Multiple Levels:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if 'n_samples' not in k else f"  {k}: {v}")
```

---

## 3. EVENT-WINDOW METRICS

### 3.1. Định nghĩa

**Event-Window MAE:**
```
Tính MAE trong cửa sổ xung quanh event:
- t-w_before đến t+w_after

Cho phép đánh giá:
- Model có anticipate event không (before)
- Model có react đúng không (after)
```

### 3.2. Implementation

```python
def event_window_mae(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     event_indices: np.ndarray,
                     window_before: int = 2,
                     window_after: int = 5) -> Dict:
    """
    MAE trong event windows
    
    Args:
        event_indices: Indices của event days
        window_before: Số ngày trước event
        window_after: Số ngày sau event
    
    Returns:
        Dict với:
        - mae_pre: MAE trong window trước event
        - mae_event: MAE tại event day
        - mae_post: MAE trong window sau event
        - mae_full_window: MAE toàn bộ window
    """
    n = len(y_true)
    
    pre_errors = []
    event_errors = []
    post_errors = []
    
    for event_idx in event_indices:
        # Pre-event window
        pre_start = max(0, event_idx - window_before)
        for i in range(pre_start, event_idx):
            if i < n:
                pre_errors.append(np.abs(y_true[i] - y_pred[i]))
        
        # Event day
        if event_idx < n:
            event_errors.append(np.abs(y_true[event_idx] - y_pred[event_idx]))
        
        # Post-event window
        post_end = min(n, event_idx + window_after + 1)
        for i in range(event_idx + 1, post_end):
            post_errors.append(np.abs(y_true[i] - y_pred[i]))
    
    return {
        'mae_pre_event': np.mean(pre_errors) if pre_errors else np.nan,
        'mae_event_day': np.mean(event_errors) if event_errors else np.nan,
        'mae_post_event': np.mean(post_errors) if post_errors else np.nan,
        'mae_full_window': np.mean(pre_errors + event_errors + post_errors),
        'n_events': len(event_indices),
        'n_pre_samples': len(pre_errors),
        'n_post_samples': len(post_errors)
    }


def event_window_analysis(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          event_indices: np.ndarray,
                          window_size: int = 5) -> Dict:
    """
    Detailed analysis: MAE tại mỗi offset từ event
    
    Returns:
        Dict với MAE tại offset -5, -4, ..., 0, ..., +4, +5
    """
    n = len(y_true)
    
    # Collect errors for each offset
    offset_errors = {offset: [] for offset in range(-window_size, window_size + 1)}
    
    for event_idx in event_indices:
        for offset in range(-window_size, window_size + 1):
            idx = event_idx + offset
            if 0 <= idx < n:
                error = np.abs(y_true[idx] - y_pred[idx])
                offset_errors[offset].append(error)
    
    # Compute mean for each offset
    results = {}
    for offset, errors in offset_errors.items():
        if errors:
            results[f"offset_{offset:+d}"] = np.mean(errors)
        else:
            results[f"offset_{offset:+d}"] = np.nan
    
    return results


def visualize_event_window_performance(event_analysis: Dict):
    """Visualize MAE across event window"""
    import matplotlib.pyplot as plt
    
    offsets = []
    maes = []
    
    for key, value in sorted(event_analysis.items()):
        if key.startswith('offset_'):
            offset = int(key.replace('offset_', '').replace('+', ''))
            offsets.append(offset)
            maes.append(value)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(offsets, maes, color=['red' if o == 0 else 'blue' for o in offsets])
    
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Event Day')
    ax.axhline(y=np.nanmean(maes), color='green', linestyle='--', alpha=0.5, label='Mean MAE')
    
    ax.set_xlabel('Days from Event')
    ax.set_ylabel('MAE')
    ax.set_title('Model Performance Across Event Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

---

## 4. DIRECTIONAL ACCURACY TRONG TAIL

### 4.1. Định nghĩa

**Directional Accuracy:**
```
DA = % predictions có đúng direction (tăng/giảm)

Standard DA: Tính trên TẤT CẢ samples
Tail DA: Chỉ tính trên EXTREME samples

Tail DA quan trọng hơn vì:
- Extreme days: Direction sai → Loss lớn
- Normal days: Direction sai → Loss nhỏ
```

### 4.2. Implementation

```python
def directional_accuracy(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         threshold: float = 0) -> float:
    """
    Basic directional accuracy
    
    Args:
        threshold: Minimum change để count (filter noise)
    """
    if len(y_true) < 2:
        return np.nan
    
    # Direction of actual changes
    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
    
    # Apply threshold
    if threshold > 0:
        significant = np.abs(y_true[1:] - y_true[:-1]) > threshold
        true_direction = true_direction[significant]
        pred_direction = pred_direction[significant]
    
    if len(true_direction) == 0:
        return np.nan
    
    correct = (true_direction == pred_direction)
    return np.mean(correct)


def tail_directional_accuracy(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              returns: np.ndarray,
                              k_percent: float = 0.1) -> Dict:
    """
    Directional accuracy chỉ trên extreme days
    
    Returns:
        Dict với:
        - da_overall: DA trên tất cả
        - da_tail: DA trên top k% extreme days
        - da_positive_tail: DA trên positive extreme days
        - da_negative_tail: DA trên negative extreme days
    """
    # Identify extreme days
    abs_returns = np.abs(returns)
    threshold = np.quantile(abs_returns, 1 - k_percent)
    
    extreme_mask = abs_returns >= threshold
    positive_extreme = (returns >= threshold)
    negative_extreme = (returns <= -threshold)
    
    # Direction changes
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Adjust mask for diff (one less element)
    extreme_mask = extreme_mask[1:]
    positive_extreme = positive_extreme[1:]
    negative_extreme = negative_extreme[1:]
    
    results = {
        'da_overall': np.mean(true_direction == pred_direction)
    }
    
    # Tail DA
    if extreme_mask.sum() > 0:
        results['da_tail'] = np.mean(
            true_direction[extreme_mask] == pred_direction[extreme_mask]
        )
    else:
        results['da_tail'] = np.nan
    
    # Positive tail DA
    if positive_extreme.sum() > 0:
        results['da_positive_tail'] = np.mean(
            true_direction[positive_extreme] == pred_direction[positive_extreme]
        )
    else:
        results['da_positive_tail'] = np.nan
    
    # Negative tail DA  
    if negative_extreme.sum() > 0:
        results['da_negative_tail'] = np.mean(
            true_direction[negative_extreme] == pred_direction[negative_extreme]
        )
    else:
        results['da_negative_tail'] = np.nan
    
    results['n_tail_samples'] = extreme_mask.sum()
    
    return results


def up_down_capture_ratio(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          returns: np.ndarray,
                          k_percent: float = 0.1) -> Dict:
    """
    Up/Down capture ratio trong tails
    
    Metrics từ portfolio management:
    - Up capture: Khi market tăng mạnh, model capture bao nhiêu %
    - Down capture: Khi market giảm mạnh, model dự đoán giảm bao nhiêu %
    """
    threshold = np.quantile(np.abs(returns), 1 - k_percent)
    
    # Up days (large positive returns)
    up_mask = returns >= threshold
    if up_mask.sum() > 0:
        up_capture = np.sum(y_pred[up_mask]) / np.sum(y_true[up_mask])
    else:
        up_capture = np.nan
    
    # Down days (large negative returns)
    down_mask = returns <= -threshold
    if down_mask.sum() > 0:
        down_capture = np.sum(y_pred[down_mask]) / np.sum(y_true[down_mask])
    else:
        down_capture = np.nan
    
    return {
        'up_capture_ratio': up_capture,
        'down_capture_ratio': down_capture,
        'capture_spread': up_capture - down_capture if not np.isnan(up_capture) and not np.isnan(down_capture) else np.nan
    }
```

---

## 5. SHOCK RECALL VÀ PRECISION

### 5.1. Framing as Classification

**Ý tưởng:** Coi shock detection như classification problem.

```
Actual shock: |return| > threshold
Predicted shock: |prediction| > threshold hoặc |error| > threshold

Metrics:
- Precision: Khi model "báo động", có đúng không?
- Recall: Model có "bắt được" bao nhiêu % shocks thật?
- F1: Harmonic mean của Precision và Recall
```

### 5.2. Implementation

```python
def shock_detection_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           returns: np.ndarray,
                           shock_threshold: float = None,
                           pred_threshold_multiplier: float = 0.5) -> Dict:
    """
    Shock detection as classification
    
    Args:
        shock_threshold: Threshold cho actual shock (nếu None, dùng 2σ)
        pred_threshold_multiplier: Prediction threshold = actual × multiplier
    
    Returns:
        Dict với precision, recall, f1 cho shock detection
    """
    # Define shock threshold
    if shock_threshold is None:
        shock_threshold = 2 * np.std(returns)
    
    pred_threshold = shock_threshold * pred_threshold_multiplier
    
    # Binary labels
    actual_shock = np.abs(returns) > shock_threshold
    predicted_shock = np.abs(y_pred - y_true) > pred_threshold
    
    # Alternative: Predicted shock khi prediction lớn
    # predicted_shock = np.abs(y_pred) > np.quantile(np.abs(y_pred), 0.9)
    
    # Confusion matrix elements
    tp = np.sum(actual_shock & predicted_shock)
    fp = np.sum(~actual_shock & predicted_shock)
    fn = np.sum(actual_shock & ~predicted_shock)
    tn = np.sum(~actual_shock & ~predicted_shock)
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'shock_precision': precision,
        'shock_recall': recall,
        'shock_f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'n_actual_shocks': actual_shock.sum(),
        'n_predicted_shocks': predicted_shock.sum(),
        'shock_threshold': shock_threshold
    }


def extreme_movement_detection(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               returns: np.ndarray,
                               k_percent: float = 0.05) -> Dict:
    """
    Detect extreme movements (top k% moves)
    
    Different from shock detection:
    - Shock: Based on threshold
    - Extreme: Based on percentile (always k% of data)
    """
    threshold = np.quantile(np.abs(returns), 1 - k_percent)
    
    # Actual extreme days
    actual_extreme = np.abs(returns) >= threshold
    
    # Predicted extreme: model's largest predicted changes
    pred_changes = np.abs(y_pred - np.roll(y_pred, 1))
    pred_changes[0] = 0
    pred_threshold = np.quantile(pred_changes, 1 - k_percent)
    predicted_extreme = pred_changes >= pred_threshold
    
    # Metrics
    tp = np.sum(actual_extreme & predicted_extreme)
    fp = np.sum(~actual_extreme & predicted_extreme)
    fn = np.sum(actual_extreme & ~predicted_extreme)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'extreme_precision': precision,
        'extreme_recall': recall,
        'extreme_f1': f1,
        'detection_rate': recall,  # Same as recall
        'false_alarm_rate': fp / (~actual_extreme).sum() if (~actual_extreme).sum() > 0 else 0
    }


def early_warning_score(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        returns: np.ndarray,
                        k_percent: float = 0.05,
                        lead_time: int = 1) -> Dict:
    """
    Can model predict extreme movements BEFORE they happen?
    
    Args:
        lead_time: Số ngày trước để check prediction
    
    Returns:
        Dict với early warning metrics
    """
    threshold = np.quantile(np.abs(returns), 1 - k_percent)
    
    # Extreme days
    actual_extreme = np.abs(returns) >= threshold
    extreme_indices = np.where(actual_extreme)[0]
    
    # Check if model "warned" lead_time days before
    warnings_before_extreme = 0
    total_extremes = 0
    
    for idx in extreme_indices:
        if idx >= lead_time:
            total_extremes += 1
            
            # Check prediction lead_time days before
            pred_change = np.abs(y_pred[idx - lead_time] - y_pred[idx - lead_time - 1]) if idx > lead_time else 0
            pred_threshold = np.quantile(np.abs(np.diff(y_pred)), 0.9)
            
            if pred_change > pred_threshold:
                warnings_before_extreme += 1
    
    early_warning_rate = warnings_before_extreme / total_extremes if total_extremes > 0 else 0
    
    return {
        'early_warning_rate': early_warning_rate,
        'total_extreme_events': total_extremes,
        'early_warnings_given': warnings_before_extreme,
        'lead_time_days': lead_time
    }
```

---

## 6. COMPREHENSIVE EVALUATION FRAMEWORK

### 6.1. All-in-One Evaluator

```python
class TailMetricsEvaluator:
    """
    Comprehensive evaluation framework cho tail-aware metrics
    """
    
    def __init__(self, 
                 tail_percentiles: List[float] = [0.05, 0.1, 0.2],
                 event_window: int = 5):
        self.tail_percentiles = tail_percentiles
        self.event_window = event_window
    
    def evaluate(self, 
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 returns: np.ndarray,
                 event_indices: Optional[np.ndarray] = None) -> Dict:
        """
        Run all tail-aware evaluations
        
        Returns:
            Comprehensive dict of all metrics
        """
        results = {}
        
        # 1. Overall metrics
        results['overall'] = {
            'mae': np.mean(np.abs(y_true - y_pred)),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mape': np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100,
            'n_samples': len(y_true)
        }
        
        # 2. Tail metrics at multiple levels
        for k in self.tail_percentiles:
            level_key = f"tail_{int(k*100)}pct"
            results[level_key] = {
                'mae': tail_mae(y_true, y_pred, returns, k),
                'rmse': tail_rmse(y_true, y_pred, returns, k),
            }
        
        # 3. Directional accuracy
        da_results = tail_directional_accuracy(y_true, y_pred, returns, k_percent=0.1)
        results['directional'] = da_results
        
        # 4. Shock detection
        results['shock_detection'] = shock_detection_metrics(y_true, y_pred, returns)
        
        # 5. Event window metrics (if events provided)
        if event_indices is not None and len(event_indices) > 0:
            results['event_window'] = event_window_mae(
                y_true, y_pred, event_indices,
                window_before=self.event_window,
                window_after=self.event_window
            )
        
        # 6. Early warning
        results['early_warning'] = early_warning_score(y_true, y_pred, returns)
        
        return results
    
    def print_report(self, results: Dict):
        """Pretty print evaluation report"""
        print("\n" + "="*70)
        print("TAIL-AWARE EVALUATION REPORT")
        print("="*70)
        
        # Overall
        print("\n--- OVERALL METRICS ---")
        overall = results['overall']
        print(f"  MAE:  {overall['mae']:.4f}")
        print(f"  RMSE: {overall['rmse']:.4f}")
        print(f"  MAPE: {overall['mape']:.2f}%")
        print(f"  N:    {overall['n_samples']}")
        
        # Tail metrics
        print("\n--- TAIL METRICS ---")
        for k in self.tail_percentiles:
            level_key = f"tail_{int(k*100)}pct"
            if level_key in results:
                print(f"  Top {int(k*100)}% extreme days:")
                print(f"    MAE:  {results[level_key]['mae']:.4f}")
                print(f"    RMSE: {results[level_key]['rmse']:.4f}")
        
        # Directional
        print("\n--- DIRECTIONAL ACCURACY ---")
        da = results['directional']
        print(f"  Overall DA:        {da['da_overall']*100:.1f}%")
        print(f"  Tail DA (10%):     {da['da_tail']*100:.1f}%")
        print(f"  Positive Tail DA:  {da.get('da_positive_tail', 0)*100:.1f}%")
        print(f"  Negative Tail DA:  {da.get('da_negative_tail', 0)*100:.1f}%")
        
        # Shock detection
        print("\n--- SHOCK DETECTION ---")
        shock = results['shock_detection']
        print(f"  Precision: {shock['shock_precision']*100:.1f}%")
        print(f"  Recall:    {shock['shock_recall']*100:.1f}%")
        print(f"  F1 Score:  {shock['shock_f1']*100:.1f}%")
        print(f"  Actual shocks: {shock['n_actual_shocks']}")
        
        # Event window
        if 'event_window' in results:
            print("\n--- EVENT WINDOW PERFORMANCE ---")
            ew = results['event_window']
            print(f"  Pre-event MAE:  {ew['mae_pre_event']:.4f}")
            print(f"  Event day MAE:  {ew['mae_event_day']:.4f}")
            print(f"  Post-event MAE: {ew['mae_post_event']:.4f}")
        
        # Early warning
        print("\n--- EARLY WARNING ---")
        ew = results['early_warning']
        print(f"  Early warning rate: {ew['early_warning_rate']*100:.1f}%")
        print(f"  Lead time: {ew['lead_time_days']} days")
        
        print("\n" + "="*70)
    
    def compare_models(self, 
                       models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models_results: Dict of {model_name: results_dict}
        
        Returns:
            DataFrame comparing all models
        """
        comparison = []
        
        for model_name, results in models_results.items():
            row = {'model': model_name}
            
            # Overall
            row['mae_overall'] = results['overall']['mae']
            row['rmse_overall'] = results['overall']['rmse']
            
            # Tail
            for k in self.tail_percentiles:
                level_key = f"tail_{int(k*100)}pct"
                if level_key in results:
                    row[f'mae_{level_key}'] = results[level_key]['mae']
            
            # Directional
            row['da_overall'] = results['directional']['da_overall']
            row['da_tail'] = results['directional']['da_tail']
            
            # Shock
            row['shock_f1'] = results['shock_detection']['shock_f1']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        return df.set_index('model')
```

### 6.2. Usage Example

```python
# Initialize evaluator
evaluator = TailMetricsEvaluator(
    tail_percentiles=[0.05, 0.1, 0.2],
    event_window=5
)

# Evaluate single model
results = evaluator.evaluate(
    y_true=y_test,
    y_pred=predictions,
    returns=returns_test,
    event_indices=event_days
)

# Print report
evaluator.print_report(results)

# Compare multiple models
models_results = {
    'Standard_MSE': evaluator.evaluate(y_test, pred_standard, returns_test),
    'Tail_Aware': evaluator.evaluate(y_test, pred_tail, returns_test),
    'Combined': evaluator.evaluate(y_test, pred_combined, returns_test)
}

comparison_df = evaluator.compare_models(models_results)
print("\nModel Comparison:")
print(comparison_df.to_string())
```

---

## 7. IMPLEMENTATION

### 7.1. Integration với sklearn

```python
from sklearn.metrics import make_scorer

# Create sklearn-compatible scorers
def tail_mae_scorer(y_true, y_pred, returns=None, k=0.1):
    """Wrapper for sklearn"""
    if returns is None:
        returns = y_true  # Fallback
    return -tail_mae(y_true, y_pred, returns, k)  # Negative because sklearn maximizes

# Register as scorer
tail_scorer = make_scorer(
    tail_mae_scorer,
    greater_is_better=False,
    needs_proba=False
)
```

### 7.2. Visualization

```python
import matplotlib.pyplot as plt

def visualize_tail_performance(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               returns: np.ndarray,
                               k_percent: float = 0.1):
    """Visualize prediction performance trên tails"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot: Actual vs Predicted
    ax1 = axes[0, 0]
    threshold = np.quantile(np.abs(returns), 1 - k_percent)
    tail_mask = np.abs(returns) >= threshold
    
    ax1.scatter(y_true[~tail_mask], y_pred[~tail_mask], alpha=0.3, label='Normal', s=10)
    ax1.scatter(y_true[tail_mask], y_pred[tail_mask], alpha=0.7, color='red', label='Tail', s=30)
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Actual vs Predicted (Tails Highlighted)')
    ax1.legend()
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    errors_normal = np.abs(y_true[~tail_mask] - y_pred[~tail_mask])
    errors_tail = np.abs(y_true[tail_mask] - y_pred[tail_mask])
    
    ax2.hist(errors_normal, bins=30, alpha=0.5, label=f'Normal (MAE={np.mean(errors_normal):.4f})')
    ax2.hist(errors_tail, bins=30, alpha=0.5, label=f'Tail (MAE={np.mean(errors_tail):.4f})')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')
    ax2.legend()
    
    # 3. MAE by return quantile
    ax3 = axes[1, 0]
    quantiles = np.arange(0, 1.05, 0.05)
    mae_by_quantile = []
    
    for i in range(len(quantiles) - 1):
        q_low = np.quantile(np.abs(returns), quantiles[i])
        q_high = np.quantile(np.abs(returns), quantiles[i + 1])
        mask = (np.abs(returns) >= q_low) & (np.abs(returns) < q_high)
        if mask.sum() > 0:
            mae_by_quantile.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        else:
            mae_by_quantile.append(np.nan)
    
    ax3.bar(range(len(mae_by_quantile)), mae_by_quantile)
    ax3.axvline(x=len(mae_by_quantile) * (1 - k_percent), color='red', linestyle='--', 
                label=f'Tail threshold ({int(k_percent*100)}%)')
    ax3.set_xlabel('Return Quantile')
    ax3.set_ylabel('MAE')
    ax3.set_title('MAE by Return Magnitude')
    ax3.legend()
    
    # 4. Cumulative error
    ax4 = axes[1, 1]
    sorted_idx = np.argsort(np.abs(returns))[::-1]
    cumulative_mae = np.cumsum(np.abs(y_true[sorted_idx] - y_pred[sorted_idx])) / np.arange(1, len(y_true) + 1)
    
    ax4.plot(np.arange(len(y_true)) / len(y_true) * 100, cumulative_mae)
    ax4.axvline(x=k_percent * 100, color='red', linestyle='--', label=f'Top {int(k_percent*100)}%')
    ax4.set_xlabel('Top % by |Return|')
    ax4.set_ylabel('Cumulative MAE')
    ax4.set_title('Cumulative MAE (sorted by |Return|)')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
```

---

## 8. BÀI TẬP THỰC HÀNH

### Bài tập 1: Implement Basic Tail Metrics

**Yêu cầu:**
1. Implement tail_mae và tail_rmse
2. Test trên synthetic data
3. Verify correctness

### Bài tập 2: Shock Detection

**Yêu cầu:**
1. Implement shock_detection_metrics
2. Tune threshold cho FPT data
3. Analyze precision-recall trade-off

### Bài tập 3: Model Comparison

**Yêu cầu:**
1. Train 3 models với different losses
2. Evaluate với TailMetricsEvaluator
3. Visualize và compare

### Bài tập 4: Event Window Analysis

**Yêu cầu:**
1. Identify earnings events trong data
2. Compute event window metrics
3. Visualize performance around events

---

## Kiểm tra hiểu bài

- [ ] Giải thích được tại sao tail metrics quan trọng
- [ ] Implement được Tail-MAE và Tail-RMSE
- [ ] Tính được directional accuracy trong tails
- [ ] Implement được shock recall/precision
- [ ] Sử dụng được comprehensive evaluator

---

## Tài liệu tham khảo

**Papers:**
- "Evaluating Financial Forecasting Models" - various
- "Performance Measures for Tail Risk Models" - Ziegel (2016)

**Related:**
- sklearn metrics documentation
- Quantile regression evaluation

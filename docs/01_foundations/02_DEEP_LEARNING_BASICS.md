# 🧠 DEEP LEARNING CƠ BẢN
## Neural Networks từ đầu - Hiểu để làm

---

## 📚 MỤC LỤC

1. [Neural Network là gì?](#1-neural-network-là-gì)
2. [Perceptron - Neuron đơn giản](#2-perceptron---neuron-đơn-giản)
3. [Activation Functions](#3-activation-functions)
4. [Forward Propagation](#4-forward-propagation)
5. [Loss Functions](#5-loss-functions)
6. [Backpropagation](#6-backpropagation)
7. [Gradient Descent](#7-gradient-descent)
8. [Overfitting & Regularization](#8-overfitting--regularization)
9. [Bài tập thực hành](#9-bài-tập-thực-hành)

---

## 1. NEURAL NETWORK LÀ GÌ?

### 🧠 Lấy cảm hứng từ não bộ

**Não người:**
```
Neuron (tế bào thần kinh):
- Nhận tín hiệu từ nhiều neurons khác
- Xử lý tín hiệu
- Gửi tín hiệu đến neurons tiếp theo

Ví dụ: Nhận diện mặt người
Input → Neurons nhận diện cạnh → Neurons nhận diện hình dạng → Neurons nhận diện khuôn mặt → Output
```

**Neural Network (mô phỏng):**
```
Artificial Neuron:
- Nhận inputs (x1, x2, x3, ...)
- Tính tổng có trọng số: w1*x1 + w2*x2 + w3*x3 + b
- Áp dụng activation function
- Gửi output đến layer tiếp theo
```

### 📊 Kiến trúc cơ bản

```
Input Layer → Hidden Layer(s) → Output Layer

Ví dụ dự đoán giá FPT:

Input:          Hidden:         Output:
close ○         ○               
ma20  ○    →    ○     →         ○ price_tomorrow
rsi   ○         ○               
macd  ○         ○               
```

### 🎯 Tại sao gọi là "Deep" Learning?

**Shallow (Nông):**
```
Input → 1 Hidden Layer → Output
→ Học được patterns đơn giản
```

**Deep (Sâu):**
```
Input → Hidden 1 → Hidden 2 → Hidden 3 → ... → Output
→ Học được patterns phức tạp, hierarchical
```

**Ví dụ:**
- Layer 1: Học low-level features (giá tăng/giảm)
- Layer 2: Học mid-level features (xu hướng ngắn hạn)
- Layer 3: Học high-level features (regime, patterns phức tạp)

---

## 2. PERCEPTRON - NEURON ĐƠN GIẢN

### 🔬 Perceptron là gì?

> **Perceptron = Neural network đơn giản nhất (1 neuron)**

### 📐 Công thức

```
y = f(w1*x1 + w2*x2 + ... + wn*xn + b)
    ↑  ↑                              ↑
    f  weights                        bias
```

**Giải thích:**
- `x1, x2, ..., xn`: Inputs (features)
- `w1, w2, ..., wn`: Weights (trọng số)
- `b`: Bias (độ lệch)
- `f`: Activation function
- `y`: Output

### 🎯 Ví dụ cụ thể

**Bài toán:** Dự đoán FPT tăng (1) hay giảm (0)

**Inputs:**
```
x1 = close = 100
x2 = ma20 = 95
x3 = rsi = 65
```

**Weights (ban đầu random):**
```
w1 = 0.5
w2 = -0.3
w3 = 0.8
b = -10
```

**Tính toán:**
```
Step 1: Weighted sum
z = w1*x1 + w2*x2 + w3*x3 + b
  = 0.5*100 + (-0.3)*95 + 0.8*65 + (-10)
  = 50 - 28.5 + 52 - 10
  = 63.5

Step 2: Activation (Sigmoid)
y = sigmoid(z) = 1 / (1 + e^(-63.5))
  ≈ 1.0

Step 3: Kết luận
y ≈ 1 → Dự đoán TĂNG
```

### 💡 Ý nghĩa của Weights

**Weight lớn (|w| cao):**
- Feature quan trọng
- Ảnh hưởng mạnh đến output

**Weight nhỏ (|w| thấp):**
- Feature ít quan trọng
- Ảnh hưởng yếu đến output

**Weight dương (+):**
- Feature tăng → Output tăng

**Weight âm (-):**
- Feature tăng → Output giảm

**Ví dụ:**
```
w_rsi = 0.8 (lớn, dương)
→ RSI tăng → Xác suất tăng giá cao

w_ma20 = -0.3 (nhỏ, âm)
→ MA20 tăng → Xác suất tăng giá giảm nhẹ
```

---

## 3. ACTIVATION FUNCTIONS

### 🤔 Tại sao cần Activation Function?

**Không có activation:**
```
y = w1*x1 + w2*x2 + b
→ Chỉ là linear function
→ Không học được patterns phức tạp
```

**Có activation:**
```
y = f(w1*x1 + w2*x2 + b)
→ Non-linear function
→ Học được patterns phức tạp
```

### 📊 Các loại Activation Functions

#### **1. Sigmoid**

**Công thức:**
```
σ(x) = 1 / (1 + e^(-x))
```

**Đồ thị:**
```
  1.0 ┤        ╭────
      │      ╱
  0.5 ┤    ╱
      │  ╱
  0.0 ┤╱
      └────────────→ x
     -∞    0    +∞
```

**Đặc điểm:**
- Output: (0, 1)
- Dùng cho: Binary classification, output layer
- Ưu: Dễ interpret (xác suất)
- Nhược: Vanishing gradient problem

#### **2. Tanh (Hyperbolic Tangent)**

**Công thức:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Đồ thị:**
```
  1.0 ┤       ╭────
      │     ╱
  0.0 ┤   ╱
      │ ╱
 -1.0 ┤╱
      └────────────→ x
     -∞    0    +∞
```

**Đặc điểm:**
- Output: (-1, 1)
- Dùng cho: Hidden layers
- Ưu: Zero-centered (tốt hơn Sigmoid)
- Nhược: Vẫn có vanishing gradient

#### **3. ReLU (Rectified Linear Unit)**

**Công thức:**
```
ReLU(x) = max(0, x)
```

**Đồ thị:**
```
      │      ╱
      │    ╱
      │  ╱
  0.0 ┤╱
      └────────────→ x
     -∞    0    +∞
```

**Đặc điểm:**
- Output: [0, +∞)
- Dùng cho: Hidden layers (phổ biến nhất)
- Ưu: Nhanh, không vanishing gradient
- Nhược: Dying ReLU problem

#### **4. Leaky ReLU**

**Công thức:**
```
LeakyReLU(x) = max(0.01*x, x)
```

**Đồ thị:**
```
      │      ╱
      │    ╱
      │  ╱
  0.0 ┤╱
      ╱
     ╱────────────→ x
    -∞    0    +∞
```

**Đặc điểm:**
- Output: (-∞, +∞)
- Dùng cho: Hidden layers
- Ưu: Giải quyết dying ReLU
- Nhược: Thêm hyperparameter (alpha)

### 💡 Chọn Activation nào?

| Layer | Activation | Lý do |
|-------|-----------|-------|
| **Hidden layers** | ReLU hoặc Leaky ReLU | Nhanh, hiệu quả |
| **Output (Regression)** | Linear (không activation) | Output không bị giới hạn |
| **Output (Binary)** | Sigmoid | Output là xác suất (0-1) |
| **Output (Multi-class)** | Softmax | Output là phân phối xác suất |

---

## 4. FORWARD PROPAGATION

### 🔄 Forward Propagation là gì?

> **Forward Propagation = Tính toán từ input → output**

### 📊 Ví dụ cụ thể

**Network:**
```
Input (2 features) → Hidden (3 neurons) → Output (1 neuron)

x1 ○     ○ h1
     →   ○ h2  →  ○ y
x2 ○     ○ h3
```

**Step-by-step:**

**Step 1: Input → Hidden**
```python
# Inputs
x = [100, 95]  # [close, ma20]

# Weights (Input → Hidden)
W1 = [[0.5, -0.3, 0.8],   # weights cho x1
      [0.2,  0.6, -0.4]]  # weights cho x2
b1 = [-10, 5, 2]          # biases

# Tính z1 (weighted sum)
z1 = x @ W1 + b1
   = [100, 95] @ [[0.5, -0.3, 0.8],
                  [0.2,  0.6, -0.4]] + [-10, 5, 2]
   = [50+19, -30+57, 80-38] + [-10, 5, 2]
   = [69, 27, 42] + [-10, 5, 2]
   = [59, 32, 44]

# Áp dụng activation (ReLU)
h = ReLU(z1)
  = [59, 32, 44]  # Tất cả > 0 nên giữ nguyên
```

**Step 2: Hidden → Output**
```python
# Weights (Hidden → Output)
W2 = [[0.7],
      [-0.5],
      [0.9]]
b2 = [3]

# Tính z2
z2 = h @ W2 + b2
   = [59, 32, 44] @ [[0.7], [-0.5], [0.9]] + [3]
   = [59*0.7 + 32*(-0.5) + 44*0.9] + [3]
   = [41.3 - 16 + 39.6] + [3]
   = [64.9] + [3]
   = [67.9]

# Áp dụng activation (Linear cho regression)
y = z2
  = 67.9

→ Dự đoán giá ngày mai: 67.9 (nghìn đồng)
```

### 🔧 Code Implementation

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def forward_propagation(x, W1, b1, W2, b2):
    """
    Forward pass through network
    
    Args:
        x: Input (shape: [batch_size, input_dim])
        W1: Weights layer 1 (shape: [input_dim, hidden_dim])
        b1: Bias layer 1 (shape: [hidden_dim])
        W2: Weights layer 2 (shape: [hidden_dim, output_dim])
        b2: Bias layer 2 (shape: [output_dim])
    
    Returns:
        y: Output predictions
        cache: Intermediate values for backprop
    """
    # Layer 1: Input → Hidden
    z1 = x @ W1 + b1
    h = relu(z1)
    
    # Layer 2: Hidden → Output
    z2 = h @ W2 + b2
    y = z2  # Linear activation for regression
    
    # Cache for backprop
    cache = {'x': x, 'z1': z1, 'h': h, 'z2': z2}
    
    return y, cache
```

---

## 5. LOSS FUNCTIONS

### 🎯 Loss Function là gì?

> **Loss Function = Đo lường "sai" bao nhiêu**

**Mục đích:**
- Đo khoảng cách giữa prediction và actual
- Càng nhỏ càng tốt
- Dùng để update weights

### 📊 Các loại Loss Functions

#### **1. MSE (Mean Squared Error)**

**Công thức:**
```
MSE = (1/n) × Σ(y_true - y_pred)²
```

**Khi nào dùng:**
- Regression problems
- Muốn phạt nặng outliers

**Code:**
```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

#### **2. MAE (Mean Absolute Error)**

**Công thức:**
```
MAE = (1/n) × Σ|y_true - y_pred|
```

**Khi nào dùng:**
- Regression problems
- Không muốn phạt nặng outliers

**Code:**
```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

#### **3. Binary Cross-Entropy**

**Công thức:**
```
BCE = -(1/n) × Σ[y_true × log(y_pred) + (1-y_true) × log(1-y_pred)]
```

**Khi nào dùng:**
- Binary classification (tăng/giảm)

**Code:**
```python
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7  # Tránh log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### 💡 Chọn Loss nào?

| Task | Loss Function |
|------|---------------|
| **Regression (giá cổ phiếu)** | MSE hoặc MAE |
| **Binary Classification (tăng/giảm)** | Binary Cross-Entropy |
| **Multi-class Classification** | Categorical Cross-Entropy |

---

## 6. BACKPROPAGATION

### 🔄 Backpropagation là gì?

> **Backpropagation = Tính gradient của loss theo từng weight**

**Mục đích:**
- Biết được weight nào cần tăng/giảm
- Biết được tăng/giảm bao nhiêu
- Dùng để update weights

### 📐 Chain Rule

**Ý tưởng:**
```
∂Loss/∂W1 = ∂Loss/∂y × ∂y/∂z2 × ∂z2/∂h × ∂h/∂z1 × ∂z1/∂W1
            ↑         ↑         ↑         ↑         ↑
         Output    Linear    ReLU     Linear    Input
```

### 🎯 Ví dụ đơn giản

**Network:**
```
x → [W, b] → z → ReLU → h → [W2, b2] → y
```

**Forward:**
```
x = 2
W = 0.5
b = 1
z = W*x + b = 0.5*2 + 1 = 2
h = ReLU(z) = 2
W2 = 0.8
b2 = 0.5
y = W2*h + b2 = 0.8*2 + 0.5 = 2.1

y_true = 3
Loss = (y_true - y)² = (3 - 2.1)² = 0.81
```

**Backward:**
```
∂Loss/∂y = -2(y_true - y) = -2(3 - 2.1) = -1.8

∂y/∂W2 = h = 2
∂Loss/∂W2 = ∂Loss/∂y × ∂y/∂W2 = -1.8 × 2 = -3.6

∂y/∂h = W2 = 0.8
∂h/∂z = 1 (vì z > 0, ReLU derivative = 1)
∂z/∂W = x = 2
∂Loss/∂W = ∂Loss/∂y × ∂y/∂h × ∂h/∂z × ∂z/∂W
         = -1.8 × 0.8 × 1 × 2
         = -2.88
```

### 🔧 Code Implementation

```python
def backward_propagation(cache, y_true, y_pred, W2):
    """
    Backward pass to compute gradients
    
    Args:
        cache: Intermediate values from forward pass
        y_true: True labels
        y_pred: Predictions
        W2: Weights layer 2
    
    Returns:
        grads: Dictionary of gradients
    """
    x = cache['x']
    z1 = cache['z1']
    h = cache['h']
    
    # Gradient of loss w.r.t. output
    dL_dy = 2 * (y_pred - y_true) / len(y_true)
    
    # Gradient w.r.t. W2, b2
    dL_dW2 = h.T @ dL_dy
    dL_db2 = np.sum(dL_dy, axis=0)
    
    # Gradient w.r.t. h
    dL_dh = dL_dy @ W2.T
    
    # Gradient w.r.t. z1 (ReLU derivative)
    dL_dz1 = dL_dh * (z1 > 0)
    
    # Gradient w.r.t. W1, b1
    dL_dW1 = x.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0)
    
    grads = {
        'dW1': dL_dW1,
        'db1': dL_db1,
        'dW2': dL_dW2,
        'db2': dL_db2
    }
    
    return grads
```

---

## 7. GRADIENT DESCENT

### 🎯 Gradient Descent là gì?

> **Gradient Descent = Thuật toán update weights để giảm loss**

**Ý tưởng:**
```
1. Tính gradient (hướng tăng loss)
2. Đi ngược hướng gradient (để giảm loss)
3. Lặp lại cho đến khi loss không giảm nữa
```

### 📊 Visualize

```
Loss
  ↑
  │     ╱╲
  │    ╱  ╲
  │   ╱    ╲
  │  ╱      ╲
  │ ╱        ╲___
  │╱             ╲
  └────────────────→ Weight
  
  Start here ●
  ↓ (gradient descent)
  ↓
  ↓
  End here (minimum) ●
```

### 🔧 Công thức

```
W_new = W_old - learning_rate × gradient

Ví dụ:
W_old = 0.5
gradient = -2.88
learning_rate = 0.01

W_new = 0.5 - 0.01 × (-2.88)
      = 0.5 + 0.0288
      = 0.5288
```

### 💡 Learning Rate

**Learning rate quá lớn:**
```
Loss
  ↑
  │     ╱╲
  │    ╱  ╲
  │   ●────●  ← Nhảy qua lại, không hội tụ
  │  ╱      ╲
  │ ╱        ╲
  └────────────→ Weight
```

**Learning rate quá nhỏ:**
```
Loss
  ↑
  │     ╱╲
  │    ╱  ╲
  │   ●→→→→  ← Chậm, mất nhiều thời gian
  │  ╱      ╲
  │ ╱        ╲
  └────────────→ Weight
```

**Learning rate vừa phải:**
```
Loss
  ↑
  │     ╱╲
  │    ╱  ╲
  │   ●→→●  ← Nhanh và hội tụ
  │  ╱      ╲
  │ ╱        ╲
  └────────────→ Weight
```

### 🔧 Code Implementation

```python
def gradient_descent_step(W1, b1, W2, b2, grads, learning_rate):
    """
    Update weights using gradient descent
    
    Args:
        W1, b1, W2, b2: Current weights
        grads: Gradients from backprop
        learning_rate: Step size
    
    Returns:
        Updated weights
    """
    W1 = W1 - learning_rate * grads['dW1']
    b1 = b1 - learning_rate * grads['db1']
    W2 = W2 - learning_rate * grads['dW2']
    b2 = b2 - learning_rate * grads['db2']
    
    return W1, b1, W2, b2
```

### 📊 Training Loop

```python
def train(X_train, y_train, epochs=100, learning_rate=0.01):
    """
    Full training loop
    """
    # Initialize weights randomly
    W1 = np.random.randn(X_train.shape[1], 10) * 0.01
    b1 = np.zeros(10)
    W2 = np.random.randn(10, 1) * 0.01
    b2 = np.zeros(1)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred, cache = forward_propagation(X_train, W1, b1, W2, b2)
        
        # Compute loss
        loss = mse_loss(y_train, y_pred)
        losses.append(loss)
        
        # Backward pass
        grads = backward_propagation(cache, y_train, y_pred, W2)
        
        # Update weights
        W1, b1, W2, b2 = gradient_descent_step(W1, b1, W2, b2, grads, learning_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return W1, b1, W2, b2, losses
```

---

## 8. OVERFITTING & REGULARIZATION

### 🎯 Overfitting trong Neural Networks

**Dấu hiệu:**
- Training loss rất thấp
- Validation loss cao
- Model "nhớ" training data thay vì "học" pattern

### 💡 Các kỹ thuật chống Overfitting

#### **1. L2 Regularization (Weight Decay)**

**Ý tưởng:**
- Phạt weights lớn
- Ép weights nhỏ lại

**Công thức:**
```
Loss_total = Loss_data + λ × Σ(W²)
                         ↑
                    Regularization term
```

**Code:**
```python
def mse_loss_with_l2(y_true, y_pred, W1, W2, lambda_reg=0.01):
    data_loss = np.mean((y_true - y_pred) ** 2)
    reg_loss = lambda_reg * (np.sum(W1**2) + np.sum(W2**2))
    return data_loss + reg_loss
```

#### **2. Dropout**

**Ý tưởng:**
- Randomly "tắt" một số neurons trong training
- Ép network học robust features

**Visualize:**
```
Training:
x ○     ○ h1 (active)
     →  ✗ h2 (dropped)  →  ○ y
x ○     ○ h3 (active)

Testing:
x ○     ○ h1
     →  ○ h2  →  ○ y
x ○     ○ h3
```

**Code:**
```python
def dropout(h, dropout_rate=0.5, training=True):
    if training:
        mask = np.random.rand(*h.shape) > dropout_rate
        return h * mask / (1 - dropout_rate)
    else:
        return h
```

#### **3. Early Stopping**

**Ý tưởng:**
- Dừng training khi validation loss không giảm nữa

**Code:**
```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(epochs):
    # Training...
    val_loss = evaluate(X_val, y_val)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

---

## 9. BÀI TẬP THỰC HÀNH

### 🎯 Bài tập 1: Implement Perceptron

**Đề bài:**
Implement perceptron từ đầu để dự đoán FPT tăng/giảm

**Gợi ý:**
```python
class Perceptron:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim) * 0.01
        self.b = 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        z = X @ self.W + self.b
        return self.sigmoid(z)
    
    def train(self, X, y, epochs=100, lr=0.01):
        # TODO: Implement training loop
        pass
```

**Kiểm tra:**
- [ ] Implement được forward pass
- [ ] Tính được loss
- [ ] Implement được backward pass
- [ ] Train được model

---

### 🎯 Bài tập 2: Build 2-Layer Network

**Đề bài:**
Build neural network 2 layers để dự đoán giá FPT

**Gợi ý:**
```python
class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
    
    def forward(self, X):
        # TODO: Implement
        pass
    
    def backward(self, X, y):
        # TODO: Implement
        pass
    
    def train(self, X, y, epochs=100, lr=0.01):
        # TODO: Implement
        pass
```

**Kiểm tra:**
- [ ] Implement được forward pass
- [ ] Implement được backward pass
- [ ] Train được model
- [ ] So sánh với Linear Regression

---

## ✅ KIỂM TRA HIỂU BÀI

Trước khi sang bài tiếp theo, hãy đảm bảo bạn:

- [ ] Giải thích được neural network là gì
- [ ] Hiểu được perceptron và cách hoạt động
- [ ] Liệt kê được các activation functions và khi nào dùng
- [ ] Hiểu được forward propagation
- [ ] Hiểu được backpropagation và chain rule
- [ ] Implement được gradient descent
- [ ] Hiểu được overfitting và cách khắc phục
- [ ] Làm được 2 bài tập thực hành

**Nếu chưa pass hết checklist, đọc lại phần tương ứng!**

---

## 📚 TÀI LIỆU THAM KHẢO

**Videos:**
- 3Blue1Brown: Neural Networks series
- Andrew Ng: Deep Learning Specialization

**Books:**
- "Deep Learning" - Goodfellow, Bengio, Courville
- "Neural Networks and Deep Learning" - Michael Nielsen

---

## 🚀 BƯỚC TIẾP THEO

Sau khi hoàn thành bài này, sang:
- `02_modeling/03_LSTM_GRU.md` - LSTM cho time series

**Chúc bạn học tốt! 🎓**

import torch

# Reproducibility
torch.manual_seed(42)

# ===== 1) Generate synthetic data y = w*x + b + noise =====
N = 200
true_w = 2.5
true_b = -0.8

X = torch.randn(N, 1)                           # shape: [N, 1]
noise = 0.3 * torch.randn(N, 1)
y = true_w * X + true_b + noise                 # target

# Train/val split
idx = torch.randperm(N)
train_size = int(0.8 * N)
train_idx, val_idx = idx[:train_size], idx[train_size:]
X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]

# Device (CPU is fine for this)
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

# ===== 2) Parameters (we learn these) =====
w = torch.randn(1, requires_grad=True, device=device)  # scalar weight
b = torch.zeros(1, requires_grad=True, device=device)  # scalar bias

# ===== 3) Training loop (manual SGD with autograd) =====
lr = 0.1
epochs = 400

for epoch in range(1, epochs + 1):
    # Forward
    y_pred = X_train * w + b
    loss = ((y_pred - y_train) ** 2).mean()

    # Backward
    loss.backward()

    # Gradient step (no_grad because we're updating leaf tensors)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Zero gradients for next step
    w.grad.zero_()
    b.grad.zero_()

    # Occasionally report validation loss
    if epoch % 50 == 0 or epoch == 1:
        with torch.no_grad():
            val_loss = ((X_val * w + b - y_val) ** 2).mean()
        print(f"Epoch {epoch:3d} | train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f} | w={w.item():.3f} | b={b.item():.3f}")

print("\nTrue params:     w=%.3f  b=%.3f" % (true_w, true_b))
print("Learned params:  w=%.3f  b=%.3f" % (w.item(), b.item()))

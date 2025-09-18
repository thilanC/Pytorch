import torch
import torch.nn as nn

torch.manual_seed(42)

# Data
N = 200
true_w = 2.5
true_b = -0.8
X = torch.randn(N, 1)
noise = 0.3 * torch.randn(N, 1)
y = true_w * X + true_b + noise

idx = torch.randperm(N)
train_size = int(0.8 * N)
train_idx, val_idx = idx[:train_size], idx[train_size:]
X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]

device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

# Model: single linear layer y = Wx + b
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

model = LinearModel().to(device)

# Loss + Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 300
for epoch in range(1, epochs + 1):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
            W = model.linear.weight.item()
            b = model.linear.bias.item()
        print(f"Epoch {epoch:3d} | train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f} | W={W:.3f} | b={b:.3f}")

W = model.linear.weight.item()
b = model.linear.bias.item()
print("\nTrue params:     w=%.3f  b=%.3f" % (true_w, true_b))
print("Learned params:  w=%.3f  b=%.3f" % (W, b))

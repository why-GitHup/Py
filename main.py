import torch
import torch.nn as nn

# Gradient Descent Algorithm from scratch


# 10 data points
N = 10

D_in = 1
D_out = 1

X = torch.randn(N, D_in)

true_W = torch.tensor([[2.0]])
true_B = torch.tensor(1.0)

y_true = X@true_W + true_B + torch.randn(N, D_out)*0.1

learning_rate, epochs = 0.01, 100

W, b = torch.randn(1, 1, requires_grad=True), torch.randn(1, requires_grad=True)

#training Loop
for epoch in range(epochs):
    #forward pass and loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true)**2)

    #backward pass
    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad; b-= learning_rate * b.grad

    #Zero Paramerters
    W.grad.zero_(); b.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}, W={W.item():.3f}, b={b.item():.3f}")

   

print(f"final parameters: W ={W.item():.3f}, b={b.item():.3f}")
print(f"True Parameters: W=2.000, b=1.000")
import numpy as np
from ann.cost_functions import compute_loss, compute_derivative_loss
from ann.lr_decay import LR_SCHEDULES


def forward_pass(X, layers, mode='train'):
    A = X
    for layer in layers:
        import inspect
        sig = inspect.signature(layer.forward)
        if 'mode' in sig.parameters:
            A = layer.forward(A, mode=mode)
        else:
            A = layer.forward(A)
    return A

def backward_pass(y_pred, y_true, layers, cost_type):
    dA = compute_derivative_loss(y_pred, y_true, cost_type)

    for layer in reversed(layers):
        dA = layer.backward(dA)


def train(X, Y, layers, epochs, learning_rate, cost_type,
          lr_decay='constant', print_every=10, **decay_kwargs):

    if lr_decay not in LR_SCHEDULES:
        raise ValueError(f"Unknown lr_decay '{lr_decay}'. "
                         f"Available: {list(LR_SCHEDULES)}")

    decay_fn = LR_SCHEDULES[lr_decay]
    lr_0     = learning_rate
    m        = X.shape[0]
    loss_history = []

    for epoch in range(1, epochs + 1):

        lr = decay_fn(epoch, lr_0, **decay_kwargs)

        y_pred = forward_pass(X, layers, mode='train')

        loss = compute_loss(y_pred, Y, cost_type) / m
        loss_history.append(loss)

        backward_pass(y_pred, Y, layers, cost_type)

        for layer in layers:
            layer.update(lr, m, epoch)

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  |  "
                  f"lr = {lr:.6f}  |  loss = {loss:.6f}")

    return loss_history


def predict(X, layers):
    return forward_pass(X, layers, mode='test')

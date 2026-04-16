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


# DODATO: Funkcija sada prima X_val i Y_val kao opcione argumente
def train(X, Y, layers, epochs, learning_rate, cost_type,
          lr_decay='constant', print_every=10, batch_size=1024, 
          X_val=None, Y_val=None, **decay_kwargs): # Izmena u zaglavlju

    if lr_decay not in LR_SCHEDULES:
        raise ValueError(f"Unknown lr_decay '{lr_decay}'. "
                         f"Available: {list(LR_SCHEDULES)}")

    decay_fn = LR_SCHEDULES[lr_decay]
    lr_0     = learning_rate
    m        = X.shape[0]
    
    # DODATO: Koristimo rečnik da čuvamo i trening i validacioni loss
    history = {'loss': [], 'val_loss': []} 
    
    global_step = 1

    for epoch in range(1, epochs + 1):

        lr = decay_fn(epoch, lr_0, **decay_kwargs)

        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        epoch_losses = []

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]
            batch_m = X_batch.shape[0]

            y_pred = forward_pass(X_batch, layers, mode='train')

            loss = compute_loss(y_pred, Y_batch, cost_type) / batch_m
            epoch_losses.append(loss)

            backward_pass(y_pred, Y_batch, layers, cost_type)

            for layer in layers:
                layer.update(lr, batch_m, global_step)
            
            global_step += 1

        avg_epoch_loss = np.mean(epoch_losses)
        history['loss'].append(avg_epoch_loss)

        # --- DODATO: Evaluacija na validacionom skupu (ako je prosleđen) ---
        val_loss_str = ""
        if X_val is not None and Y_val is not None:
            # Važno: mode='test' kako se ne bi primenjivao dropout (ako ga imate) na validaciji
            y_val_pred = forward_pass(X_val, layers, mode='test') 
            # Delimo sa Y_val.shape[0] da dobijemo prosečan gubitak (mean loss), 
            # isto kao što to radite i za batch_m iznad
            avg_val_loss = compute_loss(y_val_pred, Y_val, cost_type) / Y_val.shape[0]
            history['val_loss'].append(avg_val_loss)
            val_loss_str = f"  |  val_loss = {avg_val_loss:.6f}"
        # ---------------------------------------------------------------

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  |  "
                  f"lr = {lr:.6f}  |  loss = {avg_epoch_loss:.6f}{val_loss_str}") # Dodat ispis

    return history # Vraćamo rečnik umesto liste


def predict(X, layers):
    return forward_pass(X, layers, mode='test')
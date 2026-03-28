import numpy as np
from dense      import Dense
from dropout    import Dropout
from batch_norm import BatchNorm
from train      import train, predict
from helper     import one_hot, standardize, accuracy

np.random.seed(42)


def load_iris_split():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    X_tr_s, mu, std = standardize(X_tr)
    X_te_s, _,  _   = standardize(X_te, mu, std)
    return X_tr_s, X_te_s, y_tr, y_te


def demo_xor():
    print("=" * 62)
    print("  Demo 1 — XOR")
    print("=" * 62)

    X     = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_int = np.array([0, 1, 1, 0])
    Y     = one_hot(y_int, 2)

    layers = [
        Dense(2, 4, 'relu'),
        Dense(4, 2, 'softmax'),
    ]
    for l in layers: print(" ", l)
    print()

    loss_history = train(X, Y, layers,
                         epochs=5000, learning_rate=0.1,
                         cost_type='cross-entropy',
                         lr_decay='constant', print_every=1000)

    y_pred = predict(X, layers)
    print(f"\nFinal loss : {loss_history[-1]:.6f}")
    print(f"Accuracy   : {accuracy(y_pred, y_int)*100:.1f}%")
    print("\nPredictions:")
    for xi, yi, pi in zip(X, y_int, y_pred):
        print(f"  {xi.astype(int)}  true={yi}  pred={np.argmax(pi)}  "
              f"proba={pi.round(4)}")


def demo_iris_plain():
    print("\n" + "=" * 62)
    print("  Demo 2 — Iris  (plain, no regularisation)")
    print("=" * 62)

    X_tr, X_te, y_tr, y_te = load_iris_split()
    Y_tr = one_hot(y_tr, 3)

    layers = [
        Dense(4, 16, 'relu'),
        Dense(16, 8, 'relu'),
        Dense(8,  3, 'softmax'),
    ]
    for l in layers: print(" ", l)
    print()

    loss_history = train(X_tr, Y_tr, layers,
                         epochs=2000, learning_rate=0.05,
                         cost_type='cross-entropy',
                         lr_decay='constant', print_every=400)

    print(f"\nTrain accuracy : {accuracy(predict(X_tr, layers), y_tr)*100:.2f}%")
    print(f"Test  accuracy : {accuracy(predict(X_te, layers), y_te)*100:.2f}%")


def demo_iris_dropout():
    print("\n" + "=" * 62)
    print("  Demo 3 — Iris  (with Dropout  p=0.8)")
    print("=" * 62)

    X_tr, X_te, y_tr, y_te = load_iris_split()
    Y_tr = one_hot(y_tr, 3)

    layers = [
        Dense(4,  16, 'relu'),
        Dropout(p=0.8),
        Dense(16,  8, 'relu'),
        Dropout(p=0.8),
        Dense(8,   3, 'softmax'),
    ]
    for l in layers: print(" ", l)
    print()

    loss_history = train(X_tr, Y_tr, layers,
                         epochs=2000, learning_rate=0.05,
                         cost_type='cross-entropy',
                         lr_decay='constant', print_every=400)

    print(f"\nTrain accuracy : {accuracy(predict(X_tr, layers), y_tr)*100:.2f}%")
    print(f"Test  accuracy : {accuracy(predict(X_te, layers), y_te)*100:.2f}%")


def demo_iris_batchnorm():
    print("\n" + "=" * 62)
    print("  Demo 4 — Iris  (with Batch Normalisation)")
    print("=" * 62)

    X_tr, X_te, y_tr, y_te = load_iris_split()
    Y_tr = one_hot(y_tr, 3)

    layers = [
        Dense(4,  16, 'linear'), 
        BatchNorm(),
        Dense(16,  8, 'relu'),
        BatchNorm(),
        Dense(8,   3, 'softmax'),
    ]
    for l in layers: print(" ", l)
    print()

    loss_history = train(X_tr, Y_tr, layers,
                         epochs=2000, learning_rate=0.05,
                         cost_type='cross-entropy',
                         lr_decay='step_decay', print_every=400,
                         F=0.5, D=500)

    print(f"\nTrain accuracy : {accuracy(predict(X_tr, layers), y_tr)*100:.2f}%")
    print(f"Test  accuracy : {accuracy(predict(X_te, layers), y_te)*100:.2f}%")


if __name__ == '__main__':
    demo_xor()
    demo_iris_plain()
    demo_iris_dropout()
    demo_iris_batchnorm()

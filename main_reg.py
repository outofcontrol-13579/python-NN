import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from dime12.architecture.neural_net import *
from dime12.solver import Solver
plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)

with open('vision/code/nn/dime12/test/data_ref2.pkl', 'rb') as file:
  X, y = pickle.load(file)

print(f"Dataset shapes — X: {X.shape}, y: {y.shape}")

data = {}
data['X_train'] = X
data['X_val'] = X
data['y_train'] = y
data['y_val'] = y

learning_rate = 0.2
model = NeuralNetwork(
    [100, 50],
    # normalization='layernorm',
    dtype=np.float64
)

solver = Solver(
    model,
    data,
    batch_size=None,
    num_epochs=20000,
    print_every=1000,
    num_train_samples=None,
    update_rule="sgd",
    optim_config={"learning_rate": learning_rate},
)
solver.train()

plt.plot(solver.loss_history)
plt.title("Training loss history")
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.grid(linestyle='--', linewidth=0.5)


def nature(vec):
  return (np.sin(vec[0] * 10) / vec[0] / 10) * (np.sin(vec[1] * 10) / vec[1] / 10)


# ── if P=2, plot the reality and the model predictions ─────────────────────────────────
fig3d, axes3d = plt.subplots(subplot_kw=dict(projection="3d"))
max_x = 1.1
gran = max_x / 100
valsx = np.arange(-max_x, max_x, gran)
valsy = valsx
x_grid, y_grid = np.meshgrid(valsx, valsy)
valsall = np.array([x_grid.flatten(), y_grid.flatten()])

reality = nature(valsall).reshape(x_grid.shape)
axes3d.plot_surface(x_grid, y_grid, reality, label="reality", color="green", alpha=0.2)
axes3d.scatter(X[:, 0], X[:, 1], y, s=4, edgecolors="black", label="observations")

yhat = model.loss(valsall.T).reshape(x_grid.shape)
axes3d.plot_surface(x_grid, y_grid, yhat, label="model", color="red", alpha=0.4)
axes3d.set(xlabel="x", ylabel="y", title=f"Two-hidden-layer ReLU NN  (data loss {solver.loss_history[-1]:.6e})")
axes3d.legend()

fig3d, axes3d = plt.subplots(subplot_kw=dict(projection="3d"))
diff = yhat - reality
axes3d.plot_surface(x_grid, y_grid, diff, label="diff", color="orange", alpha=0.2)
axes3d.legend()
plt.show()

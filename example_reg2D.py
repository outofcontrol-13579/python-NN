# To visualize overfitting: manually change line 311 in solver.train() so that solver stores model parameters with best training losses, not validation losses

import numpy as np
import matplotlib.pyplot as plt
from dime12.architecture.neural_net import *
from dime12.architecture.lin_reg import *
from dime12.solver import Solver
from dime12.datasets.generate_data import *
plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)

# ── generate data ────────────────────────────────────
data = generate_2D_data(num_train=100, num_val=100)
print(
    f"Dataset shapes — X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}, X_val: {data['X_val'].shape}, y_val: {data['y_val'].shape}")

# ── regress linearly ─────────────────────────────────
df, betahat, stderrs, tstats, pvals, confintervs, rse, Rsquared, F, rss, residuals, l2_loss = linreg(
    data['y_train'], data['X_train_expanded']
)

print('****************** Linear Model *****************************************************')
print(df)
print('rse:                             ', rse)
print('R-squared:                       ', Rsquared)
print(f"\nl2 training loss:                 {l2_loss:.6e}")
y_valhat = np.c_[np.ones(data['X_val_expanded'].shape[0]), data['X_val_expanded']] @ betahat
residuals_val = data['y_val'] - y_valhat
l2_loss_val = 0.5 * np.dot(residuals_val, residuals_val) / data['X_val_expanded'].shape[0]
print(f"\nl2 validation loss:               {l2_loss_val:.6e}")
print('*************************************************************************************')
print("\n")

# ── train a neural network ────────────────────────────
print('******************* Neural Network **************************************************')
learning_rate = 2e-5
model = NeuralNetwork(
    [100, 100, 100, 100, 100],
    input_dim=2,
    # normalization='layernorm',
    dtype=np.float64)

solver = Solver(
    model,
    data,
    batch_size=None,
    num_epochs=20000,
    print_every=1000,
    num_train_samples=None,
    update_rule="adam",
    optim_config={"learning_rate": learning_rate},
    verbose=False
)
solver.train()

print(f"\nl2 training loss:                 {solver.loss_history[-1]:.6e}")
print(f"\nl2 validation loss:               {solver.val_acc_history[-1]:.6e}")
print('*************************************************************************************')

plt.title("Loss: neural network vs. linear model")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(linestyle='--', linewidth=0.5)
plt.plot(solver.loss_history, label=f"nn training", color='blue')
plt.plot(solver.val_acc_history, label=f"nn validation", color='orange')
plt.plot(len(solver.val_acc_history), l2_loss, 'x', label=f"lin training", color='blue')
plt.plot(len(solver.val_acc_history), l2_loss_val, 'x', label=f"lin validation", color='orange')
plt.legend()

# ── if P=2, plot the truth vs. the predictions ─────────────────────────────────
fig3d, axes3d = plt.subplots(subplot_kw=dict(projection="3d"))
max_x = 1.1
gran = max_x / 100
valsx = np.arange(-max_x, max_x, gran)
valsy = valsx
x_grid, y_grid = np.meshgrid(valsx, valsy)
valsall = np.array([x_grid.flatten(), y_grid.flatten()])

truth = define_truth(valsall).reshape(x_grid.shape)
axes3d.plot_surface(x_grid, y_grid, truth, label="truth", color="green", alpha=0.2)
axes3d.scatter(data['X_train'][:, 0], data['X_train'][:, 1], data['y_train'],
               s=4, edgecolors="black", label="observations (training)")

yhat = model.loss(valsall.T).reshape(x_grid.shape)
axes3d.plot_surface(x_grid, y_grid, yhat, label="model", color="red", alpha=0.4)
axes3d.set(xlabel="x", ylabel="y",
           title=f"Two-hidden-layer ReLU NN  (training data loss {solver.loss_history[-1]:.6e})")
axes3d.legend()

fig3d, axes3d = plt.subplots(subplot_kw=dict(projection="3d"))
axes3d.plot_surface(x_grid, y_grid, truth, label="truth", color="green", alpha=0.2)
axes3d.scatter(data['X_val'][:, 0], data['X_val'][:, 1], data['y_val'],
               s=4, edgecolors="black", label="observations (validation)")
axes3d.plot_surface(x_grid, y_grid, yhat, label="model", color="red", alpha=0.4)
axes3d.set(xlabel="x", ylabel="y",
           title=f"Two-hidden-layer ReLU NN  (validation data loss {solver.val_acc_history[-1]:.6e})")
axes3d.legend()

plt.show()

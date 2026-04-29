# high-dimensional data: nn training and hyperparameter search

import time
import numpy as np
import matplotlib.pyplot as plt
from dime12.architecture.neural_net import *
from dime12.architecture.lin_reg import *
from dime12.solver import Solver
from dime12.datasets.generate_data import *
plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)

# ── generate data ────────────────────────────────────
P = 2
data = generate_data(input_dim=P, num_train=10000, num_val=1000)
print(f"Dataset shapes — "
      f"X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}, "
      f"X_val: {data['X_val'].shape}, y_val: {data['y_val'].shape}")

# ── regress linearly ─────────────────────────────────
df, betahat, stderrs, tstats, pvals, confintervs, rse, Rsquared, F, rss, residuals, l2_loss = linreg(
    data['y_train'], data['X_train'])

print('****************** Linear Model *****************************************************')
print(df)
print('rse:                             ', rse)
print('R-squared:                       ', Rsquared)
print(f"l2 training loss:                 {l2_loss:.6e}")
y_valhat = np.c_[np.ones(data['X_val'].shape[0]), data['X_val']] @ betahat
residuals_val = data['y_val'] - y_valhat
l2_loss_val = 0.5 * np.dot(residuals_val, residuals_val) / data['X_val'].shape[0]
print(f"l2 validation loss:               {l2_loss_val:.6e}")
print('*************************************************************************************')
print("\n")

# ── train neural networks - run 30 experiments ────────────────────────────
best_val = 1e6    # keep track of best validation accuracy
best_params = {}  # dict to hold best model parameters

rng = np.random.default_rng(1303)
tic = time.time()
for i in range(30):
  lr = 10 ** rng.uniform(-5, -3)   # learning rate
  reg = 10 ** rng.uniform(-3, 0)   # reguarization
  kr = rng.uniform(.7, .9)         # dropout keep ratio

  model = NeuralNetwork(
      [100, 100, 100],
      input_dim=P,
      reg=reg,
      dropout_keep_ratio=kr,
      seed=21
      #   normalization='batchnorm'
  )

  solver = Solver(
      model, data,
      batch_size=256, num_epochs=300,
      num_train_samples=None,
      update_rule="adam",
      optim_config={"learning_rate": lr},
      verbose=False
  )

  solver.train()                # train the model
  new_val = solver.best_val_acc  # extract best accuracy
 # Save if validation accuracy is the best
  if new_val < best_val:
    best_val = new_val
    best_params = {'lr': lr, 'reg': reg, 'kr': kr}
  toc = time.time()
  print(f"({toc - tic:.1f}s)")
  tic = toc

  # Print the values for each chosen hyperparameter and the validation accuracy
  print(
    f'lr: {lr:.5f} reg: {reg:.5f}, kr: {kr:.5f}, last val: {solver.val_acc_history[-1]:.5f}, best val: {new_val:.5f}')

# Print the best accuracy
print(f'Best validation accuracy: {best_val}')
print('*************************************************************************************')

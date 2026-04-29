import numpy as np
from dime12.architecture.neural_net import *
from dime12.solver import Solver
from dime12.datasets.generate_data import *
import pickle

# high-dimensional test with regularization and dropout
print('high-dimensional data, starting test')
with open('dime12/test/data_ref1.pkl', 'rb') as file:
  [data] = pickle.load(file)

print(f"Dataset shapes — "
      f"X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}, "
      f"X_val: {data['X_val'].shape}, y_val: {data['y_val'].shape}")
model = NeuralNetwork(
    [100, 100, 100],
    input_dim=data['X_train'].shape[1],
    reg=0.03405,
    dropout_keep_ratio=0.87622,
    seed=12  # this seed only for dropout
  )

solver = Solver(
    model, data,
    batch_size=256, num_epochs=3,
    num_train_samples=None,
    update_rule="adam",
    optim_config={"learning_rate": 0.00060},
    verbose=False
  )

solver.train()

with open('dime12/test/results_ref1.pkl', 'rb') as file:
  loss_history_ref, val_acc_history_ref, best_val_acc_ref = pickle.load(file)

print(loss_history_ref)
print(solver.loss_history)
if np.allclose(solver.loss_history, loss_history_ref) and  \
        np.allclose(solver.val_acc_history, val_acc_history_ref) and \
        np.allclose(solver.best_val_acc, best_val_acc_ref):
  print('high dimensional data test ok')
else:
  print('error in high dimensional data test')

# low-dimensional test without regularization and with layer normalization
print('2D data, starting test')
with open('dime12/test/data_ref2.pkl', 'rb') as file:
  X, y = pickle.load(file)

print(f"Dataset shapes — X: {X.shape}, y: {y.shape}")

data = {}
data['X_train'] = X
data['X_val'] = X
data['y_train'] = y
data['y_val'] = y

model = NeuralNetwork(
    [100, 50],
    normalization='layernorm',
    dtype=np.float64,
)

solver = Solver(
    model,
    data,
    batch_size=None,
    num_epochs=50,
    print_every=25,
    num_train_samples=None,
    update_rule="sgd_momentum",
    optim_config={"learning_rate": 1e-4},
    verbose=True
)
solver.train()

with open('dime12/test/results_ref2.pkl', 'rb') as file:
  loss_history_ref, val_acc_history_ref, best_val_acc_ref = pickle.load(file)

if np.allclose(solver.loss_history, loss_history_ref):
  print('2D test ok')
else:
  print('error in 2D test')


# # Add new test - to save:
# with open('vision/code/nn/dime12/test/losses_ref3.pkl', 'wb') as file:
#   pickle.dump([solver.loss_history, solver.val_acc_history, solver.best_val_acc], file)
# wait = input("Press Enter to continue.")

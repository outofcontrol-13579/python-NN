import numpy as np
from sklearn.datasets import make_regression


def define_truth(vec):
  # # linear function
  # return np.array([2, 3]) @ vec + 0.5
  # # quadratic function: x.T @ A @ x
  # A = np.eye(2); A[0, 1] = -0.5; A[1, 0] = 0.5
  # return np.sum(np.multiply(vec, (A @ vec)), axis=0)
  # # x1 * x2
  # return np.multiply(vec[0], vec[1])
  # # Gaussian pdf
  # mu = np.array([0, 0]); Sigma = np.array([[1, 0.3], [0.3, 0.4]]) # mu as (2, )
  # invS = np.linalg.inv(Sigma)
  # det = 1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(Sigma)))
  # xminusmu = vec - mu.reshape(-1, 1)
  # num = np.exp(-0.5 * np.sum(np.multiply(xminusmu, (invS @ xminusmu)), axis=0))
  # return num / det
  # polynomial
  beta = np.array([0.165, -0.364, -0.419, 0.23])
  # beta = np.array([1, -2, 1.1, 1])
  # X = np.array([vec[0], -vec[1], -vec[1]**3, -0.5*vec[0]**2*vec[1]]).T
  X = np.array([vec[0]**5, vec[0]**3, vec[0], vec[0] * vec[1]**2]).T
  return X @ beta
  # # sinc
  # return (np.sin(vec[0] * 10) / vec[0] / 10) * (np.sin(vec[1] * 10) / vec[1] / 10)


def generate_2D_data(num_train, num_val):  # dimensionality P = 2
  rng_data = np.random.default_rng(1303)
  max_x = 1
  gran = max_x / 100
  valsx = np.arange(-max_x, max_x, gran)
  valsy = valsx
  x_grid, y_grid = np.meshgrid(valsx, valsy)
  data = {}
  for (type, num) in zip(['train', 'val'], [num_train, num_val]):
    mask = rng_data.choice(np.arange(len(x_grid.flatten())), num)
    X = np.c_[x_grid.flatten()[mask], y_grid.flatten()[mask]]   # N x P - each row = one datapoint
    data[f'X_{type}'] = X
    data[f'X_{type}_expanded'] = np.c_[X[:, 0]**5, X[:, 0] ** 3, X[:, 0], X[:, 0] * X[:, 1]**2]
    data[f'y_{type}'] = define_truth(data[f'X_{type}'].T) + rng_data.standard_normal(num) * 0.05
  return data


def generate_data(input_dim, num_train, num_val):
  X, y = make_regression(n_samples=num_train + num_val, n_features=input_dim, n_informative=30, random_state=2)

  data = {}
  # data['X_train'] = X[:num_train, :]
  # data['X_val'] = X[num_train:, :]
  # data['y_train'] = y[:num_train]
  # data['y_val'] = y[num_train:]

  data['X_val'] = X[:num_val, :]
  data['X_train'] = X[num_val:, :]
  data['y_val'] = y[:num_val]
  data['y_train'] = y[num_val:]

  return data


# def generate_2D_data(num_train, num_val):  # dimensionality P = 2
#   rng_data = np.random.default_rng(1303)
#   max_x = 1
#   gran = max_x / 100
#   valsx = np.arange(-max_x, max_x, gran)
#   valsy = valsx
#   x_grid, y_grid = np.meshgrid(valsx, valsy)
#   valsall = np.array([x_grid.flatten(), y_grid.flatten()])
#   mask = rng_data.choice(np.arange(len(x_grid.flatten())), num_train)
#   X_train = np.c_[x_grid.flatten()[mask], y_grid.flatten()[mask]]   # N x P - each row = one datapoint
#   y_train = define_truth(X_train.T) + rng_data.standard_normal(num_train) * 0.05
#   mask_val = rng_data.choice(np.arange(len(x_grid.flatten())), num_val)
#   X_val = np.c_[x_grid.flatten()[mask_val], y_grid.flatten()[mask_val]]
#   y_val = define_truth(X_val.T) + rng_data.standard_normal(num_val) * 0.05

#   data = {}
#   data['X_train'] = X_train
#   data['X_val'] = X_val
#   data['X_train_expanded'] = np.c_[X_train[:, 0]**5, X_train[:, 0]**3, X_train[:, 0], X_train[:, 0] * X_train[:, 1]**2]
#   data['X_val_expanded'] = np.c_[X_val[:, 0]**5, X_val[:, 0]**3, X_val[:, 0], X_val[:, 0] * X_val[:, 1]**2]
#   data['y_train'] = y_train
#   data['y_val'] = y_val
#   return data

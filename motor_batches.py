from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cvxpy as cp
import pandas as pd
pd.set_option('display.float_format', '{:,.3e}'.format)
plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)
# wait = input("Press Enter to continue.")

DATASET_DIR = Path(__file__).parent / "dime12" / "datasets"
BATCH_SIZE = 100_000  # min 10_000.
FILTERTYPE = 'convolution'  # choose a filter from FILTERS below.
MOMENTUM = 0.9  # weight of the running_mean for the parameter evolution.
SHOW_PREDICTIONS = False  # plot predictions at the end of each processed measurement batch.
VERBOSE = False

START_VALUES = np.array([  # start values of the parameters R, Lqd10, etc... Use datasheet values?
  3e-02, 0., 0., 0., 0.,
  3e-02, 4e-03, 0., 0., 0., 0., 0.])
FILTERS = {
    'none': {},  # Do not filter the measurements
    'convolution': {'window': 50},  # Window: 10 for 10 samples.
    'lowess': {'span': 0.05},  # Span: 1 for the complete batch, 0.1 for one tenth of the batch.
    'total_variation_smoothing': {'delta': 0.5},  # Remove noise but keep the edges. Delta: 0 for no smoothing.
}

# LINKED TO PROBLEM STRUCTURE - DO NOT CHANGE LIGHTLY (Start)
SIGNAL_NAMES = ['Id', 'Iq', 'Ud', 'Uq', 'Wel']
PARAMETER_NAMES = [
  'R', 'Lqd10', 'Lqd30', 'Cdq01', 'Cdq11',
  'R', 'Psi', 'Cdq01', 'Ldq10', 'Ldq20', 'Ldq30', 'Cdq11']
# LINKED TO PROBLEM STRUCTURE - DO NOT CHANGE LIGHTLY (End)


def vprint(*args, **kwargs):
  if VERBOSE:
    print(*args, **kwargs)


def import_measurements(keys, dataset_dir, plot=False):
  """Load raw signals from CSV files into a dict."""
  vprint('\n*** Importing measurements ***')
  data = {}
  for key in keys:
    path = f"{dataset_dir}/{key}.csv"
    vprint(f"  importing {path}")
    data[key] = pd.read_csv(path).values
    has_nan = np.argwhere(np.isnan(data[key])).size > 0
    has_inf = np.argwhere(np.isinf(data[key])).size > 0
    vprint(f"    shape: {data[key].shape}  |  NaN: {has_nan}  Inf: {has_inf}")
  # trim as needed (for example: only keep measurements with non zero speed and/or non zero currents)
  idx_to_keep = (abs(data['Iq']) > 0.4) & (abs(data['Wel']) > 40)
  for key in keys:
    data[key] = data[key][idx_to_keep]
  # plot if wished
  if plot:
    for key in keys:
      fig, ax = plt.subplots()
      ax.plot(data[key], label=f"{key} - {len(data[key])} samples")
      ax.grid(linestyle='--', linewidth=0.5)
      ax.set_xlabel('sample')
      ax.set_ylabel('SI unit')
      plt.legend()
      fig.suptitle(f"imported measurement: {key}")
  return data, data[key].shape[0]


def build_constraints(parameter_names):
  """
    C encodes equality constraints (shared parameters): C @ x = 0.
    D encodes non-negativity constraints:               D @ x >= 0.
    """
  equality_pairs = [(0, 5), (3, 7), (4, 11)]  # x[0] == x[5], etc... since R, Cdq01, Cdq11 shared between d/q axes
  n_params = len(parameter_names)
  C = np.zeros((len(equality_pairs), n_params))
  for idx, (i, j) in enumerate(equality_pairs):
    C[idx, i] = 1
    C[idx, j] = -1
  D = np.diag([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])  # x[0] >= 0, etc... since at least R and psi should be >0
  return C, D


def extract_batch(df, signal_names, batch_slice):
  return {s: df[f'{s}'][batch_slice] for s in signal_names}


def do_filtering_in_batches(arr: np.ndarray, batch_size: int, apply_filter: callable) -> np.ndarray:
  """Apply a filter function in batches"""
  flat = arr.flatten()
  N = len(flat)
  num_batches = N // batch_size  # throw the remainder away
  all_fitted = np.empty(N)
  for i in range(num_batches):
    vprint(f"        filter batch {i}/{num_batches}")
    start = i * batch_size
    end = (i + 1) * batch_size
    all_fitted[start:end] = apply_filter(flat[start:end])
  return all_fitted


def lowess_filter(arr: np.ndarray, span: float) -> np.ndarray:
  """Apply LOWESS filter, in batches to save memory"""
  def apply(batch: np.ndarray) -> np.ndarray:
    xs = np.arange(len(batch))
    return sm.nonparametric.lowess(batch, xs, frac=span, xvals=xs)

  return do_filtering_in_batches(arr, batch_size=10000, apply_filter=apply)


def tv_filter(arr: np.ndarray, delta: float) -> np.ndarray:
  """Total variation smoothing, in batches to save memory"""
  batch_size = 5000
  D = np.eye(batch_size - 1, batch_size, k=0) * -1  # main diagonal: -x(i)
  D += np.eye(batch_size - 1, batch_size, k=1)      # superdiagonal: x(i+1)

  def apply(batch: np.ndarray) -> np.ndarray:
    x = cp.Variable(batch_size, name='x')
    obj = cp.sum_squares(x - batch) + delta * cp.norm1(D @ x)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(verbose=False, solver=cp.CLARABEL)
    return x.value

  return do_filtering_in_batches(arr, batch_size=batch_size, apply_filter=apply)


def convolution_filter(arr: np.ndarray, window: float) -> np.ndarray:
  flat = arr.flatten()
  return np.convolve(flat, np.ones(window) / window, mode='valid')


def filter_batch(df_in, signal_names, filtertype, filter_params):
  df_out = {}
  for name in signal_names:
    vprint(f"*** Filtering {name} ***")
    signal = df_in[name]
    if filtertype == 'lowess':
      filtered = lowess_filter(signal, filter_params['span'])
    elif filtertype == 'convolution':
      filtered = convolution_filter(signal, filter_params['window'])
    elif filtertype == 'total_variation_smoothing':
      filtered = tv_filter(signal, filter_params['delta'])
    elif filtertype == 'none':
      filtered = signal
    else:
      print('Specified filter is not implemented: non-filtered signal will be used.')
      filtered = signal
    # filtered16 = filtered.astype(np.float16) # if use this, do not forget to recast afterwards to avoid overflow
    df_out[name] = filtered
  return df_out


def build_problem(df, signal_names):
  # the next line is there to improve readibility but is prone to error if changed. Make sure the variable names match the signal names.
  id, iq, ud, uq, om = (df[signal] for signal in signal_names)
  num_meas = len(id)
  d_num, q_num = (5, 7)
  cols = [id, -om * iq, -om * iq**3, -om * id * iq, -0.5 * om * iq * id**2,
          iq, om, 0.5 * om * iq**2, om * id, om * id**2, om * id**3, 0.5 * om * id * iq**2,
          ]
  BE = None
  for col in cols:
    if BE is None:
      BE = col
    else:
      BE = np.c_[BE, col]
  # one block for ud, one block for uq
  Ud = BE.copy()
  Ud[:, d_num:] = np.zeros((num_meas, q_num))
  Ud = np.c_[Ud, ud]
  Uq = BE.copy()
  Uq[:, :d_num] = np.zeros((num_meas, d_num))
  Uq = np.c_[Uq, uq]
  # interleave
  BE2 = np.empty((num_meas * 2, (d_num + q_num + 1)))
  BE2[::2] = Ud
  BE2[1::2] = Uq
  return BE2[:, -1], BE2[:, :-1]


def solve_problem(b, A, C, D, start_value=None, parameter_names=None):
  x = cp.Variable(A.shape[1], name='x')
  if start_value is not None:
    x.value = start_value
  obj = cp.norm2(A @ x - b)
  constr = [
    C @ x == 0,
    D @ x >= 0,
  ]
  prob = cp.Problem(cp.Minimize(obj), constr)
  vprint('The problem is well conditioned: ', prob.is_dcp())
  try:
    prob.solve(verbose=VERBOSE, solver=cp.CLARABEL, warm_start=True)
  except cp.error.SolverError:
    print("Solver failed. Trying alternative solver...")
    # Attempt a more robust solver
    prob.solve(solver=cp.SCS, verbose=VERBOSE, max_iters=10000)
    # Check status after solving
    if prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
      print("Problem is infeasible or unbounded.")
      return pd.DataFrame(data={'coef': None}, index=parameter_names), None, None, None
    elif prob.status != cp.OPTIMAL:
      print("Solver failed to find optimal solution.")
      return pd.DataFrame(data={'coef': None}, index=parameter_names), None, None, None
  vprint("Solve time:", prob.solver_stats.solve_time)
  l2_loss = 0.5 * prob.value**2 / A.shape[0]
  df = pd.DataFrame(data={
      'coef': x.value},
      index=parameter_names)
  return df, x.value, prob.value, l2_loss


def plot_predictions(y, y_hat, df_batch, iteration):
  components = [('Ud', slice(None, None, 2)), ('Uq', slice(1, None, 2))]
  for name, sl in components:
    fig, ax = plt.subplots()
    if y_hat is None:
      fig.suptitle(f"iteration: {iteration} - solver did not converge so no predictions.")
    else:
      ax.plot(y_hat[sl], label=f"prediction {name}", color='orange', alpha=1.0)
      fig.suptitle(f"iteration: {iteration}")
    ax.plot(y[sl], label=f"data {name}", color='blue', alpha=0.4)
    ax.plot(df_batch[name], label=f"data unfiltered {name}", color='green', alpha=0.1)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.set_xlabel('sample')
    ax.set_ylabel('SI unit')
    ax.legend()
    fig.tight_layout()


def plot_parameters_evolution(params, betahats, means, batch_size):
  for param in params:
    idx = PARAMETER_NAMES.index(param)
    fig, ax = plt.subplots()
    ax.plot(means[:, idx], label=f"{param} running mean - (last={means[-1, idx]:.4g})")
    ax.plot(betahats[:, idx],
            label=f"{param} each solutions - (avg={np.mean(means[:, idx]):.4g})")
    ax.legend()
    ax.set_xlabel(f"measurement batch - each batch has {batch_size} samples")
    ax.set_ylabel('SI unit')
    ax.grid(linestyle='--', linewidth=0.5)
    ax.set_xticks(range(betahats.shape[0]))


def run_once(plot_all_meas=True, batch_size=BATCH_SIZE, filtertype=FILTERTYPE, show_predictions=SHOW_PREDICTIONS):
  # import from measurements file
  meas_all, num_meas = import_measurements(SIGNAL_NAMES, DATASET_DIR, plot=plot_all_meas)
  if num_meas < batch_size:
    print('Aborted: Not enough measurements.')
    return
  # constraints do not change so build them in advance
  C, D = build_constraints(PARAMETER_NAMES)
  # get filter parameters
  filter_params = FILTERS[filtertype]
  # start loop to iteratively analyze batches extracted from the measurements
  num_iterations = num_meas // batch_size
  running_mean = START_VALUES.copy()
  betahats = np.full((num_iterations, len(PARAMETER_NAMES)), np.nan)
  means = np.full((num_iterations, len(PARAMETER_NAMES)), np.nan)
  for i in range(num_iterations):
    print(f"************ Analyzing measurement batch {i}/{num_iterations} ************ ")
    # extract batch
    sl = slice(i * batch_size, (i + 1) * batch_size)
    meas_batch = extract_batch(meas_all, SIGNAL_NAMES, sl)
    # filter batch signals
    meas_batch_filtered = filter_batch(meas_batch, SIGNAL_NAMES, filtertype, filter_params)
    # build problem
    y, X = build_problem(meas_batch_filtered, SIGNAL_NAMES)
    # solve problem
    df_solution, betahat, prob_value, l2_loss = solve_problem(
      y, X, C, D, start_value=running_mean, parameter_names=PARAMETER_NAMES)
    betahats[i] = betahat  # can be None if solver did not converge
    running_mean = MOMENTUM * running_mean + (1 - MOMENTUM) * betahat if betahat is not None else running_mean
    means[i] = running_mean
    df_solution['running_mean'] = running_mean
    # compute and show predictions if wished
    if show_predictions:
      y_hat = X @ betahat if betahat is not None else None
      plot_predictions(y, y_hat, meas_batch, i)
      plt.show()
    # end of the loop - all batches analyzed.

  # show the evolution of some of the parameters
  print(df_solution.to_string())
  params_to_plot = ['R', 'Psi']  # 'Lqd10', 'Ldq10'
  plot_parameters_evolution(params_to_plot, betahats, means, batch_size)


def run_multiple():
  plot_all_meas = True
  for f in ['none']:
    for bs in [100_000, 50_000, 20_000]:
      print(f"batch: {bs}, filter: {f}")
      run_once(
          plot_all_meas=plot_all_meas,
          batch_size=bs,
          filtertype=f,
          show_predictions=False,
      )
      plot_all_meas = False  # only plot all measurements in the first run


if __name__ == "__main__":
  # Uncomment one of the following to run a single analysis or sweep over all filters and batch sizes:
  run_once()
  # run_multiple()
  plt.show()

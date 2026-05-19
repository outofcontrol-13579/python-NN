from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import null_space
pd.set_option('display.float_format', '{:,.3e}'.format)
plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)
# wait = input("Press Enter to continue.")

DATASET_DIR = Path(__file__).parent.parent / "dime12" / "datasets"
VERBOSE = False

# Minimum current / speed below which measurements are discarded.
IQ_THRESHOLD = 0.4
WEL_THRESHOLD = 40

# Simulate a breakdown in the machine after a first run, then add two runs with adapted measurements
BREAKDOWN = False

# Signal names must match CSV column headers exactly.
SIGNAL_NAMES = ["Id", "Iq", "Ud", "Uq", "Wel"]

# Full parameter name list (length = 12).
# Indices 0/5, 3/7, 4/11 are shared between the d and q axes.
PARAMETER_NAMES = [
    "R", "Lqd10", "Lqd30", "Cdq01", "Cdq11",                  # d-axis (cols 0-4)
    "R", "Psi", "Cdq01", "Ldq10", "Ldq20", "Ldq30", "Cdq11",  # q-axis (cols 5-11)
]
EQUALITY_PAIRS = [(0, 5), (3, 7), (4, 11)]

# Datasheet / prior-knowledge initial guesses for the parameters.
START_VALUES = np.array([
    3e-02, 50e-6, 0., 0., 0.,
    3e-02, 4e-03, 0., 50e-6, 0., 0., 0.,
])

RNG = np.random.default_rng(133)

# Regressed parameters, use to simulate broken machine.
REFERENCE_VALUES = np.array([
  3.339e-02, 7.680e-05, -6.670e-09, -2.890e-07, -1.858e-08,
  3.339e-02, 3.968e-03, -2.890e-07, 6.305e-05, -1.481e-06, -2.957e-08, -1.858e-08])


def vprint(*args, **kwargs):
  if VERBOSE:
    print(*args, **kwargs)


def import_measurements(signal_names, dataset_dir, iq_threshold=IQ_THRESHOLD, wel_threshold=WEL_THRESHOLD, plot=False, breakdown=BREAKDOWN):
  """Load raw signals from CSV files into a dict."""
  vprint('\n*** Importing measurements ***')
  data = {}
  for key in signal_names:
    path = f"{dataset_dir}/{key}.csv"
    vprint(f"  importing {path}")
    data[key] = pd.read_csv(path).values
    has_nan = np.argwhere(np.isnan(data[key])).size > 0
    has_inf = np.argwhere(np.isinf(data[key])).size > 0
    vprint(f"    shape: {data[key].shape}  |  NaN: {has_nan}  Inf: {has_inf}")

  # Discard low-excitation samples (motor near standstill or at zero load).
  keep = (np.abs(data["Iq"]) > iq_threshold) & (np.abs(data["Wel"]) > wel_threshold)
  print(f"  keeping {keep.sum()} / {len(keep)} samples after threshold filter")
  for key in signal_names:
    data[key] = data[key][keep]

  # simulate a breakdown in the machine - i.e. strong R increase or Phi decrease
  if breakdown:
    print('  simulating a breakdown in the machine after the first measurement run. Total samples: 3 *', keep.sum())
    A = expand_basis(data, signal_names)[1]
    broken_x = REFERENCE_VALUES.copy()
    # strong R increase
    broken_x[0] = 1e-1
    broken_x[5] = 1e-1
    b = A @ broken_x  # Ud, Uq
    data_half = data.copy()
    data_half['Ud'] = b[::2]
    data_half['Uq'] = b[1::2]
    for key in signal_names:
      data[key] = np.append(data[key], np.append(data_half[key], data_half[key]))

  if plot:
    for key, values in data.items():
      fig, ax = plt.subplots()
      ax.plot(values, label=f"{key} - {len(values)} samples")
      ax.grid(linestyle='--', linewidth=0.5)
      ax.set_xlabel('sample')
      ax.set_ylabel('SI unit')
      fig.suptitle(f"imported measurement: {key}")
      if breakdown:
        ax.axvline((len(values) - 1) // 3, linestyle='--', color='red', label='start breakdown')
      plt.legend()

  return data, data[key].shape[0]


def build_constraints_nullspace(n_params, equality_pairs):
  """Return F such that any x = F @ x_r satisfies all equality constraints.

  The constraint matrix C encodes x[i] == x[j] as C[k, i] = 1, C[k, j] = -1.
  F is the orthonormal nullspace of C, so C @ (F @ x_r) == 0 for all x_r.
  This reduces the free parameter count by - len(pairs).

  Parameters
    ----------
    n_params:        Total number of parameters.
    equality_pairs:  List of (i, j) index pairs that must be equal.
  """
  C = np.zeros((len(equality_pairs), n_params))
  for idx, (i, j) in enumerate(equality_pairs):
    C[idx, i] = 1
    C[idx, j] = -1
  F = null_space(C)
  return F


def expand_basis(meas, signal_names):
  """Assemble the regression targets b and design matrix (basis expansion) A for one or more samples.

  The d/q voltage equations are interleaved row-by-row:
      rows 0, 2, 4, … → d-axis equation  (Ud)
      rows 1, 3, 5, … → q-axis equation  (Uq)

  The full regression matrix has 12 columns matching PARAMETER_NAMES.
  Columns 0-4 are used only in d-axis rows (zeroed in q-axis rows),
  columns 5-11 are used only in q-axis rows (zeroed in d-axis rows).

  Parameters
  ----------
  meas:          Dict of signal arrays or scalars keyed by signal name.
  signal_names:  Ordered list ['Id', 'Iq', 'Ud', 'Uq', 'Wel'].

  Returns
  -------
  b : (2*N,)  target voltages.
  A : (2*N, 12)  basis expansion.
  """
  id, iq, ud, uq, om = (meas[signal] for signal in signal_names)
  N = len(np.atleast_1d(id))
  scalar = N == 1

  A = np.zeros((2 if scalar else N * 2, 12))

  # d-axis   R,  Lqd10,    Lqd30,       Cdq01,         Cdq11
  d_terms = [id, -om * iq, -om * iq**3, -om * id * iq, -0.5 * om * iq * id**2]
  # q-axis   R,  Psi,Cdq01,            Ldq10,   Ldq20,      Ldq30,      Cdq11
  q_terms = [iq, om, 0.5 * om * iq**2, om * id, om * id**2, om * id**3, 0.5 * om * id * iq**2]

  if scalar:
    A[0, :5] = d_terms
    A[1, 5:] = q_terms
    b = np.array([ud, uq])
  else:
    A[::2, :5] = np.column_stack(d_terms)
    A[1::2, 5:] = np.column_stack(q_terms)
    b = np.empty(N * 2)
    b[::2] = ud
    b[1::2] = uq

  return b, A


def extract_measurement(df, signal_names, idx):
  return {s: df[s][idx] for s in signal_names}


def update_solution_rank1(b_new, a_new, cache, F):
  """Rank-1 Recursive Least Squares update for a single new (b_new, a_new) observation.

    The update operates in the reduced space (columns = null-space of the constraint matrix C),
    then maps the estimate back to the full parameter space.

    Ar := A @ F, ar_new := a_new @ F

    Normal equation:
    [Arᵀ ar_new]  [ Ar        xr* = [Arᵀ ar_new] [ b
                    ar_newᵀ]                     b_new]
                    <=>
    (Arᵀ Ar + ar_new ar_newᵀ) xr* = Arᵀ b + ar_new * b_new

    Uses the Sherman-Morrison identity:
        (M + u uᵀ)⁻¹ = M⁻¹ - (M⁻¹ u)(M⁻¹ u)ᵀ / (1 + uᵀ M⁻¹ u)
    with M := Arᵀ @ Ar (PSD by construction so invertible) and u := ar_new

    Parameters
    ----------
    b_new:   New scalar target (i.e. Ud or Uq).
    a_new:   (n_params,) New regressor (i.e. basis expansion of Id, Iq and Wel) row.
    cache:   Tuple consisting of (M⁻¹, Arᵀ @ b)
    F:       (n_params, r) Null-space basis of the constraint matrix C.

    Returns
    -------
    x: updated full-space parameter estimate (mapped back from xr with x = A @ xr)
    cache: Updated cache.
    """
  old_inverse, old_rhs = cache

  # Project new regressor row into the reduced space
  ar_new = F.T @ a_new

  # Traditional Recursive Least Squares in the reduced space
  Minv_u = old_inverse @ ar_new
  denominator = 1.0 + ar_new @ Minv_u
  if np.abs(denominator) < 1e-12:
    # Observation is nearly collinear with existing data; skip the update.
    print("  Warning: near-singular rank-1 update skipped.")
    return F @ (old_inverse @ old_rhs), cache
  new_inverse = old_inverse - np.outer(Minv_u, Minv_u) / denominator  # Sherman - Morrison
  new_rhs = old_rhs + ar_new * b_new
  xr = new_inverse @ new_rhs

  # Project back into full space
  x = F @ xr
  cache = (new_inverse, new_rhs)
  return x, cache


def update_solution_rank2(b_new, A_new, cache, F):
  """Rank-2 RLS update for k new observations.

  Processes the d-axis (Ud) and q-axis (Uq) equations from one measurement
  sample together as a rank-2 update, so both rows inform the estimate
  simultaneously rather than sequentially.

  The update operates in the reduced space (r = n_params - n_constraints
  columns, spanned by F) then maps back via x = F @ xr.

  See update_solution_rank1 for the normal equation and the definitions.

  Uses the Woodbury matrix identity:
      (M + UUᵀ)⁻¹ = M⁻¹ - M⁻¹ U (I + Uᵀ M⁻¹ U)⁻¹ Uᵀ M⁻¹

  where U = Ar is of shape (r x k),  so (I + Uᵀ M⁻¹ U) is (k x k).
  For k = 2, this is a 2 x 2 matrix inversion, cheap and numerically well-behaved.
  The only division is numerically guarded.

  Parameters
  ----------
  b_new:   (2,) target voltages for this sample ([Ud, Uq]).
  A_new:   (2, n_params) regressor rows for this sample.
  cache:   Tuple consisting of (M⁻¹, Arᵀ @ b).
  F:       (n_params, r) null-space basis of the constraint matrix.

  Returns
  -------
  x : (n_params,) updated full-space parameter estimate.
  cache   : Updated RLS cache.
  """
  old_inverse, old_rhs = cache

  # Project two new regressor rows into the reduced space
  Ar_new = F.T @ A_new.T         # (r, 2)

  # Traditional Recursive Least Squares in the reduced space
  Minv_U = old_inverse @ Ar_new  # (r, 2)
  # (I + Uᵀ M⁻¹ U) is 2×2 — invert explicitly via the closed-form formula:
  #   [a b]⁻¹       1    [ d  -b]
  #   [c d]   =  ——————  [-c   a]
  #              ad - bc
  inner = np.eye(2) + Ar_new.T @ Minv_U            # (2, 2)
  a, b, c, d = inner[0, 0], inner[0, 1], inner[1, 0], inner[1, 1]
  det = a * d - b * c
  if np.abs(det) < 1e-12:
    print("  Warning: near-singular rank-2 update skipped.")
    return F @ (old_inverse @ old_rhs), cache
  inner_inv = np.array([[d, -b], [-c, a]]) / det
  # Woodbury update
  new_inverse = old_inverse - Minv_U @ inner_inv @ Minv_U.T   # (r, r)
  new_rhs = old_rhs + Ar_new @ b_new             # (r,)
  xr = new_inverse @ new_rhs

  # Project back into full space
  x = F @ xr
  return x, (new_inverse, new_rhs)


def initialise_cache(signal_names, start_values, F, n_init=50, rng=RNG):
  """Bootstrap the cache from synthetic data consistent with the machine equations.
  """

  emulated: dict[str, np.ndarray] = {
      "Id": np.linspace(-10, 10, n_init) + rng.standard_normal(n_init),
      "Iq": np.linspace(-10, 10, n_init) + rng.standard_normal(n_init),
      "Wel": np.linspace(-100, 100, n_init) + rng.standard_normal(n_init),
      "Ud": np.full(n_init, np.nan),   # will be computed from start_values
      "Uq": np.full(n_init, np.nan),
    }
  A = expand_basis(emulated, signal_names)[1]
  b = A @ start_values  # Ud, Uq
  Ar = A @ F
  cache = (np.linalg.inv(Ar.T @ Ar), Ar.T @ b)
  x = start_values
  # import pickle
  # with open('start_values.pkl', 'rb') as file:
  #   [cache, x] = pickle.load(file)
  return cache, x


def plot_parameters_evolution(param_names_to_plot, xs, breakdown=BREAKDOWN):
  """Plot convergence history for selected parameters."""
  for name in param_names_to_plot:
    idx = PARAMETER_NAMES.index(name)
    fig, ax = plt.subplots()
    ax.plot(xs[:, idx],
            label=f"{name} (avg={np.mean(xs[:, idx]):.4g})")
    ax.set_xlabel(f"measurement")
    ax.set_ylabel('SI unit')
    ax.grid(linestyle='--', linewidth=0.5)
    fig.suptitle(f"Parameter evolution: {name}")
    if breakdown:
      ax.axvline((xs.shape[0] - 1) // 3, linestyle='--', color='red', label='start breakdown')
    ax.legend()


def run_once(plot_all_meas=True, only_rank1_update=False):
  """Run the full RLS identification pipeline on all available measurements.

    Steps
    -----
    1. Load measurement CSVs.
    2. Build the equality-constraint null-space F.
    3. Initialise the RLS cache from synthetic data.
    4. Process every measurement sample one at a time (streaming RLS).
    5. Plot parameter convergence and print final estimates.

    Parameters
    ----------
    plot_all_meas:  If True, plot each imported signal before processing.

    Returns
    -------
    DataFrame with 'last' (final estimate) and 'mean' columns, indexed by
    parameter name.
    """
  # 1. Load data.
  meas_all, num_meas = import_measurements(SIGNAL_NAMES, DATASET_DIR, plot=plot_all_meas)

  # 2. Build null-space basis.
  F = build_constraints_nullspace(len(PARAMETER_NAMES), EQUALITY_PAIRS)

  # 3. Initialise start values for the cache (inverse and rhs) and x
  cache, x = initialise_cache(SIGNAL_NAMES, START_VALUES, F)

  # 4. Stream through measurements
  num_iterations = num_meas
  xs = np.full((num_iterations * 2 if only_rank1_update == 1 else num_iterations, len(PARAMETER_NAMES)), np.nan)
  for i in range(num_iterations):
    if i % 100000 == 0:
      print(f"************ Analyzing measurement [{i}/{num_iterations}] R = {x[0]:.4g} ************ ")
    new_meas = extract_measurement(meas_all, SIGNAL_NAMES, i)
    b, A = expand_basis(new_meas, SIGNAL_NAMES)
    if only_rank1_update:  # — two 1-D Recursive Least Square updates per sample (d then q axis).
      for j in range(len(b)):
        x, cache = update_solution_rank1(b[j], A[j], cache, F)
        xs[2 * i + j] = x
    else:  # Default: one 2-D Recursive Least Square updates per sample.
      x, cache = update_solution_rank2(b, A, cache, F)
      xs[i] = x
  # end of the loop - all measurements analyzed.

  # 5. Results.
  params_to_plot = ['R', 'Psi', 'Lqd10', 'Ldq10']
  plot_parameters_evolution(params_to_plot, xs)
  df_means = pd.DataFrame(data={
      'last': xs[-1],
      'mean': np.nanmean(xs, axis=0)},
      index=PARAMETER_NAMES)
  print(df_means.to_string())


if __name__ == "__main__":
  run_once()
  plt.show()

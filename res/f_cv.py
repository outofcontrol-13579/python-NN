# Regression of Id, Iq, om on Ud, Uq. See Readme.md
# CV NOTE: the 4 fitted models (bilinear/OLS, MLP, physics+residual, greybox) are evaluated with
# 5-fold cross validation. The datasheet reference has no free parameters, so it isn't cross
# validated -- it's just evaluated once on the whole dataset for context.
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import TensorDataset, DataLoader
from regressors.nets import MLP, train, Res_PSM, GreyBoxPMSM

plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)

logging.basicConfig(
    level=logging.WARNING,  # set to logging.INFO to get info-level output, set to logging.WARNING to silence
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 0

USE_GPU = False
DTYPE = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
elif USE_GPU and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Torch: running on', device)

# 5-fold CV: each fold holds out 1/5 of the (chunked) data as its test set. Of the remaining
# 4/5, VAL_FRAC is further set aside as a validation split (used for checkpoint bookkeeping /
# loss curves during training). Test folds are never touched for model selection.
N_FOLDS = 5
VAL_FRAC = 0.15

DATASET_DIR = Path(__file__).parent.parent / "dime12" / "datasets"

# Signal names must match CSV column headers exactly.
SIGNAL_NAMES = ["Id", "Iq", "Ud", "Uq", "Wel"]
PREDICTOR_KEYS = ["Id", "Iq", "Wel"]
RESPONSE_KEYS = ["Ud", "Uq"]

MOTOR_PARAMETERS = ['R', 'Ld', 'Lq', 'Psi']
DATASHEET = np.array([30e-3, 50e-6, 50e-6, 4.2e-3])


def plot_losses(train_losses, val_losses, title=""):
    fig, ax = plt.subplots()
    ax.set_title(f'Training/Validation losses - {title}')
    ax.set_xlabel('Epoch')
    ax.plot(train_losses, marker='o', label="training", color='blue')
    ax.plot(val_losses, marker='o', label="validation", color='orange')
    ax.grid(linestyle='--', linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def import_measurements(signal_names, dataset_dir, iq_threshold=0, wel_threshold=0, plot=False):
    """Load raw signals from CSV files into a dict.
    Will give a shape error if the CSV files do not all have the same number of rows.
    """
    logger.info('*** Importing measurements ***')
    data = {}
    bad_rows = None
    for key in signal_names:
        path = f"{dataset_dir}/{key}.csv"
        logger.info(f"  importing {path}")
        data[key] = pd.read_csv(path).values
        nan_mask = np.isnan(data[key]).any(axis=1)
        inf_mask = np.isinf(data[key]).any(axis=1)
        bad_mask = nan_mask | inf_mask
        n_bad = bad_mask.sum()
        logger.info(f"    shape: {data[key].shape}  |  NaN: {nan_mask.sum()}  Inf: {inf_mask.sum()}")
        if n_bad > 0:
            logger.warning(f"  WARNING: {path} has {n_bad} row(s) with NaN/Inf — will be dropped from all signals.")
        bad_rows = bad_mask if bad_rows is None else (bad_rows | bad_mask)

    if bad_rows.any():
        n_bad_total = bad_rows.sum()
        logger.warning(f"  Dropping {n_bad_total} / {len(bad_rows)} samples with NaN/Inf in any signal.")
        for key in signal_names:
            data[key] = data[key][~bad_rows]

    # Discard low-excitation samples (motor near standstill or at zero load).
    keep = (np.abs(data["Iq"]) >= iq_threshold) & (np.abs(data["Wel"]) >= wel_threshold)
    print(f"  keeping {keep.sum()} / {len(keep)} samples after threshold filter")
    for key in signal_names:
        data[key] = data[key][keep]

    if plot:
        for key, values in data.items():
            fig, ax = plt.subplots()
            ax.plot(values, label=f"{key} - {len(values)} samples")
            ax.grid(linestyle='--', linewidth=0.5)
            ax.set_xlabel('sample')
            ax.set_ylabel('SI unit')
            fig.suptitle(f"imported measurement: {key}")
            plt.legend()

    n = data[signal_names[0]].shape[0]
    return data, n


def chunked_kfold_indices(n_samples, chunk_size, n_folds=5, val_frac=0.15, seed=2):
    """Split n_samples contiguous rows into non-overlapping chunks, then assign whole chunks
    (never individual rows) to K folds, so neighboring samples -- assumed to be autocorrelated
    -- never end up split across a train/test boundary.

    For each of the n_folds folds, one group of chunks becomes that fold's test set; the
    remaining chunks are further split into train/val, with val_frac the fraction of that
    remainder (not of the total) held out for validation.

    Returns a list of n_folds dicts, each with keys 'train_idx', 'val_idx', 'test_idx' (row
    indices) and 'train_chunks', 'val_chunks', 'test_chunks' (chunk ids, for bookkeeping).
    """
    if not (0 <= val_frac < 1):
        raise ValueError(f"val_frac ({val_frac}) must be in [0, 1).")

    chunk_size = int(chunk_size)
    n_chunks = n_samples // chunk_size
    discarded = n_samples - n_chunks * chunk_size
    print(f"  discarding {discarded} / {n_samples} trailing samples "
          f"({n_chunks} full chunks of size {chunk_size}, {n_folds} folds)")

    chunk_ids = np.arange(n_chunks)

    def chunk_to_rows(chunks):
        if len(chunks) == 0:
            return np.array([], dtype=np.int64)
        idx = np.concatenate([np.arange(c * chunk_size, (c + 1) * chunk_size) for c in chunks])
        return np.sort(idx.astype(np.int64))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for trainval_chunks, test_chunks in kf.split(chunk_ids):
        if val_frac > 0:
            train_chunks, val_chunks = train_test_split(trainval_chunks, test_size=val_frac, random_state=seed)
        else:
            train_chunks, val_chunks = trainval_chunks, np.array([], dtype=int)

        folds.append({
            'train_idx': chunk_to_rows(train_chunks),
            'val_idx': chunk_to_rows(val_chunks),
            'test_idx': chunk_to_rows(test_chunks),
            'train_chunks': train_chunks,
            'val_chunks': val_chunks,
            'test_chunks': test_chunks,
        })
    return folds


def to_matrix(d, keys):
    return np.stack([d[k] for k in keys], axis=1)  # N x P


def make_loaders(X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, batch_size, seed):
    train_ds = TensorDataset(X_train_t, Y_train_t)
    val_ds = TensorDataset(X_val_t, Y_val_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)
    loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed))
    loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return loader_train, loader_val, loader_test


def get_or_train_model(
    model_path,
    model, optimizer,
    device, dtype,
    loader_train, loader_val, loader_test,
    epochs=1, stats_every=100, save=True
):
    """
    If model_path exists, load weights and skip training.
    Otherwise, train the model and save the resulting state_dict.

    Returns: model, train_loss_history, val_loss_history, test_loss
    (loss histories are None if loaded from checkpoint)
    """
    if os.path.exists(model_path):
        print(f"  Found existing checkpoint at '{model_path}' — loading, skipping training.")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"  Loaded model (saved test loss: {checkpoint.get('test_loss', 'n/a')}).")
        return model, None, None, checkpoint.get('test_loss', None)

    print(f"  No checkpoint found at '{model_path}' — training from scratch.")
    train_loss_history, val_loss_history, test_loss = train(
        device, dtype, loader_train, loader_val, loader_test,
        model, optimizer, epochs=epochs, stats_every=stats_every,
    )

    if save:
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': float(np.nanmin(val_loss_history)),
            'test_loss': float(test_loss),
            'epochs': epochs,
        }, model_path)
        print(f"Training complete — model saved to '{model_path}'.")
    else:
        print(f"Training complete — model was not saved.")

    return model, train_loss_history, val_loss_history, test_loss


def compute_regression_metrics(pred, true, response_keys):
    """RMSE and R^2 per response variable, in physical units."""
    metrics = {}
    for i, key in enumerate(response_keys):
        err = pred[:, i] - true[:, i]
        rmse = np.sqrt(np.mean(err ** 2))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        metrics[key] = {'rmse': rmse, 'r2': 1 - ss_res / ss_tot}
    return metrics


def report_regression_metrics(pred, true, response_keys, split_name=""):
    """Print RMSE/R^2 per response variable and return the same as a dict."""
    metrics = compute_regression_metrics(pred, true, response_keys)
    print(f"*** {split_name} metrics (physical units) ***")
    for key, m in metrics.items():
        print(f"  {key}: RMSE = {m['rmse']:.4f}  |  R^2 = {m['r2']:.4f}")
    return metrics


def record_fold_metrics(cv_results, model_name, metrics):
    """Append one fold's per-response RMSE/R^2 to the running cross-validation results."""
    for key, m in metrics.items():
        cv_results[model_name][key]['rmse'].append(m['rmse'])
        cv_results[model_name][key]['r2'].append(m['r2'])


def print_cv_summary(cv_results, response_keys, n_folds):
    print(f"*** {n_folds}-fold CV summary — test-fold metrics, physical units, mean +/- std ***")
    for model_name, per_key in cv_results.items():
        print(f"\n{model_name}:")
        for key in response_keys:
            rmses = np.array(per_key[key]['rmse'])
            r2s = np.array(per_key[key]['r2'])
            print(f"  {key}: RMSE = {rmses.mean():.4f} +/- {rmses.std():.4f}  |  "
                  f"R^2 = {r2s.mean():.4f} +/- {r2s.std():.4f}")


def get_predictions_mlp(model, device, dtype, X_by_split, y_scaler):
    """Run the model on each split's inputs and inverse-transform to physical units.
    X_by_split : dict mapping split name (e.g. 'train', 'val', 'test') to the corresponding input tensor.
    Returns: dict mapping split name to an (n_split, n_responses) array of physical-unit predictions.
    """
    model.eval()
    model = model.to(device)

    preds = {}
    with torch.no_grad():
        for split, X in X_by_split.items():
            pred_scaled = model(X.to(device=device, dtype=dtype)).cpu().numpy()
            preds[split] = y_scaler.inverse_transform(pred_scaled)
    return preds


def plot_predictions(preds, trues, idxs, response_keys, title=""):
    """Plot predictions against ground truth in time order, across an arbitrary
    set of named splits (e.g. 'train', 'val', 'test').

    preds, trues, idxs: dicts keyed by split name. preds[split] and trues[split]
    are (n_split, n_responses) physical-unit arrays; idxs[split] holds the
    original (time-ordered) row indices for that split.
    """
    split_colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green'}
    splits = list(idxs.keys())

    n_responses = len(response_keys)
    fig, axes = plt.subplots(n_responses, 1, figsize=(14, 3.5 * n_responses), sharex=True)
    fig.suptitle(title)
    if n_responses == 1:
        axes = [axes]

    all_idx = np.concatenate([idxs[s] for s in splits])
    order = np.argsort(all_idx)

    for i, (ax, key) in enumerate(zip(axes, response_keys)):
        # Ground truth: plot every split together in time order
        all_true = np.concatenate([trues[s][:, i] for s in splits])
        ax.plot(all_idx[order], all_true[order], color='black', lw=0.8,
                label='ground truth', alpha=0.6, zorder=1)

        # Predictions: scatter, colored by split
        for s in splits:
            ax.scatter(idxs[s], preds[s][:, i], s=4, color=split_colors.get(s, 'tab:gray'),
                       label=f'pred ({s})', alpha=0.6, zorder=2)

        ax.set_ylabel(key)
        ax.legend(loc='upper right', markerscale=2)
        ax.set_title(f'{key}: predictions vs ground truth')

    axes[-1].set_xlabel('sample index (time)')
    plt.tight_layout()

    return {'pred': preds, 'true': trues}


def build_basis_expansion(np_raw):
    """Build the interleaved (Ud/Uq) basis-expansion design matrix for a data split.

    Parameters map to [R, Ld, Lq, Psi]
    Ud rows: [id,      0, -om*iq,    0]  -> ud = R*id - Lq*om*iq
    Uq rows: [iq, om*id,       0,   om]  -> uq = R*iq + Ld*om*id + Psi*om

    Returns X (N*2, 4) and y (N*2,), interleaved row-wise as
    [Ud-row, Uq-row, Ud-row, Uq-row, ...].
    """
    id_, iq, om, ud, uq = (np_raw[signal] for signal in ['Id', 'Iq', 'Wel', 'Ud', 'Uq'])
    num_meas = len(id_)
    zeros = np.zeros(num_meas)

    Ud_row = np.column_stack([id_, zeros, -om * iq, zeros])
    Uq_row = np.column_stack([iq, om * id_, zeros, om])

    X = np.empty((num_meas * 2, 4))
    X[::2] = Ud_row
    X[1::2] = Uq_row

    y = np.empty(num_meas * 2)
    y[::2] = ud
    y[1::2] = uq

    return X, y


def get_predictions_bil(betahat, X_be):
    """Compute (Ud, Uq) predictions in physical units for each split.

    betahat : fitted parameter vector 
    X_be : dict mapping split name (e.g. 'train', 'val', 'test') to the
           predictor matrix returned by build_basis_expansion (interleaved
           Ud/Uq rows)

    Returns: dict mapping split name to an (n_split, 2) array -> [Ud, Uq].
    """
    def _predict(X_split):
        pred_interleaved = X_split @ betahat
        pred_ud = pred_interleaved[0::2]
        pred_uq = pred_interleaved[1::2]
        return np.stack([pred_ud, pred_uq], axis=1)

    return {split: _predict(X) for split, X in X_be.items()}


def compute_effective_params(model, device, dtype, loader):
    """Run the trained greybox model over a loader and collect per-sample
    operating points + effective-parameter values, weights frozen."""
    model.eval()
    model = model.to(device)
    P = model.n_predictors
    keys = ['id', 'iq', 'om', 'R_eff', 'Psi_eff', 'Ld_eff', 'Lq_eff']
    out = {k: [] for k in keys}

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device=device, dtype=dtype)
            model(x)  # populates model.*_eff as a side effect
            x_raw = x[:, P:]
            out['id'].append(x_raw[:, 0].cpu())
            out['iq'].append(x_raw[:, 1].cpu())
            out['om'].append(x_raw[:, 2].cpu())
            out['R_eff'].append(model.R_eff.cpu())
            out['Psi_eff'].append(model.Psi_eff.cpu())
            out['Ld_eff'].append(model.Ld_eff.cpu())
            out['Lq_eff'].append(model.Lq_eff.cpu())

    return {k: torch.cat(v).numpy() for k, v in out.items()}


def plot_effective_params(train_data, val_data, test_data, nominal):
    """
    train_data, val_data, test_data: dicts from compute_effective_params()
    nominal: {'R': R0, 'Psi': Psi0, 'Ld': Ld0, 'Lq': Lq0} (nominal scalars, e.g.
             torch.exp(model.log_R0).item(), model.Psi0.item(), etc.)

    Per parameter (R, Psi, Ld, Lq), one row of three panels:
      [1] relative-deviation histogram, train vs val vs test overlaid -- the
          generalization check across all three splits.
      [2] effective value vs its main physical driver, test only (the
          held-out split, never used for training or model selection),
          colored by om.
      [3] correction map over (Id, Iq), test only.

    Then a dedicated figure for the Ld_eff/Psi_eff degeneracy check, also on
    the test split. Both enter uq only through om*(Ld_eff*id + Psi_eff), so
    at small |id| the model can trade one off against the other while
    leaving uq unchanged. If that's happening, Ld_eff and Psi_eff deviations
    should be anti-correlated, and that anti-correlation should strengthen
    as 1/|id| grows.
    """
    params = [
        ('R', 'R_eff', 'iq', 'Iq [A]'),
        ('Psi', 'Psi_eff', 'id', 'Id [A]'),
        ('Ld', 'Ld_eff', 'id', 'Id [A]'),
        ('Lq', 'Lq_eff', 'iq', 'Iq [A]'),
    ]

    fig, axes = plt.subplots(4, 3)

    for row, (name, key, xkey, xlabel) in enumerate(params):
        p0 = nominal[name]
        dev_train = train_data[key] / p0 - 1.0
        dev_val = val_data[key] / p0 - 1.0
        dev_test = test_data[key] / p0 - 1.0

        # [1] Saturation / generalization check -- train + val + test overlaid
        ax = axes[row, 0]
        ax.hist(dev_train, bins=60, alpha=0.5, label='train', color='tab:blue')
        ax.hist(dev_val, bins=60, alpha=0.5, label='val', color='tab:orange')
        ax.hist(dev_test, bins=60, alpha=0.5, label='test', color='tab:green')
        ax.set_title(f'{key} / {name}0 - 1')
        ax.legend()
        ax.grid(linestyle='--', linewidth=0.5)

        # [2] Effective value vs main physical driver -- test only
        ax = axes[row, 1]
        sc = ax.scatter(test_data[xkey], test_data[key], c=test_data['om'], s=4, cmap='coolwarm', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        fig.colorbar(sc, ax=ax, label='om [rad/s]')

        # [3] Correction map over (Id, Iq) -- test only
        ax = axes[row, 2]
        sc2 = ax.scatter(test_data['id'], test_data['iq'], c=test_data[key], s=4, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Id [A]')
        ax.set_ylabel('Iq [A]')
        fig.colorbar(sc2, ax=ax, label=key)

    plt.tight_layout()

    # --- Ld_eff / Psi_eff degeneracy check, test only ---
    Ld_dev = test_data['Ld_eff'] / nominal['Ld'] - 1.0
    Psi_dev = test_data['Psi_eff'] / nominal['Psi'] - 1.0
    abs_id = np.abs(test_data['id'])
    inv_abs_id = 1.0 / np.clip(abs_id, np.percentile(abs_id, 5), None)  # clip near-zero id for a sane colorbar
    r = np.corrcoef(Ld_dev, Psi_dev)[0, 1]

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))

    sc3 = axes2[0].scatter(Ld_dev, Psi_dev, c=inv_abs_id, s=4, cmap='plasma', alpha=0.6)
    axes2[0].set_xlabel('Ld_eff / Ld0 - 1')
    axes2[0].set_ylabel('Psi_eff / Psi0 - 1')
    axes2[0].set_title(f'Ld/Psi degeneracy check (test, overall r = {r:.3f})')
    fig2.colorbar(sc3, ax=axes2[0], label='1 / |Id|  (clipped)')

    # Correlation within bins of |Id|: does anti-correlation strengthen as Id -> 0?
    n_bins = 8
    bin_edges = np.quantile(abs_id, np.linspace(0, 1, n_bins + 1))
    bin_r, bin_inv_id = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (abs_id >= lo) & (abs_id <= hi)
        if mask.sum() > 10:
            bin_r.append(np.corrcoef(Ld_dev[mask], Psi_dev[mask])[0, 1])
            bin_inv_id.append(1.0 / np.median(abs_id[mask]))
    order = np.argsort(bin_inv_id)
    bin_inv_id, bin_r = np.array(bin_inv_id)[order], np.array(bin_r)[order]

    axes2[1].plot(bin_inv_id, bin_r, marker='o')
    axes2[1].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes2[1].set_xlabel('1 / median(|Id|) per bin')
    axes2[1].set_ylabel('corr(Ld_eff, Psi_eff) within bin')
    axes2[1].set_title('Does anti-correlation strengthen as Id -> 0?')
    axes2[1].grid(linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, fig2


# --------------------------------------------------------------------------------------
# Per-model, per-fold training/prediction. Each fits on the fold's train split (with the
# fold's val split used internally for checkpoint bookkeeping / loss tracking) and returns
# predictions on the fold's held-out test split, in physical units. Checkpoint filenames
# include the fold index so folds never load / overwrite each other's weights.
# --------------------------------------------------------------------------------------

def run_bilinear(np_raw_train, np_raw_test):
    """Closed-form OLS fit of the bilinear physical model -- no NN training involved."""
    X_train_be, y_train_be = build_basis_expansion(np_raw_train)
    X_test_be, _ = build_basis_expansion(np_raw_test)
    betahat = np.linalg.pinv(X_train_be) @ y_train_be
    pred_test = get_predictions_bil(betahat, {'test': X_test_be})['test']
    motor_params = dict(zip(MOTOR_PARAMETERS, betahat.tolist()))
    return pred_test, motor_params


def run_mlp(fold_idx, X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, y_scaler, device, dtype, seed):
    arch = {'hidden_sizes': [32, 32], 'layernorms': False, 'silu': False}
    solver = {'lr': 0.00043, 'reg': 0.00357, 'batch_size': 64, 'epochs': 5}

    loader_train, loader_val, loader_test = make_loaders(
        X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, solver['batch_size'], seed
    )

    torch.manual_seed(seed)
    model = MLP(X_train_t.shape[1], arch['hidden_sizes'], Y_train_t.shape[1],
                layernorms=arch['layernorms'], silu=arch['silu'])
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': solver['reg']},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0},
    ], lr=solver['lr'])

    model_path = (
        Path(__file__).parent / "regressors" / "weights"
        / f"MLP{arch['hidden_sizes']}l{arch['layernorms']}_s{arch['silu']}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}_bs{solver['batch_size']}_ep{solver['epochs']}"
        f"_fold{fold_idx}.pth"
    )

    model, train_loss_history, val_loss_history, _ = get_or_train_model(
        model_path=model_path, model=model, optimizer=optimizer,
        device=device, dtype=dtype,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_test,
        epochs=solver['epochs'], stats_every=10, save=True,
    )
    if train_loss_history is not None:
        print(f"  [mlp]  fold {fold_idx} final loss — train: {train_loss_history[-1]:.4f}, "
              f"val: {val_loss_history[-1]:.4f}")

    return get_predictions_mlp(model, device, dtype, {'test': X_test_t}, y_scaler)['test']


def run_res(fold_idx, X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t,
            y_scaler, motor_params, device, dtype, seed):
    arch = {'hidden_sizes': [24, 24], 'lambda_prior': 0.8, 'layernorms': True, 'silu': False}
    solver = {'lr': 0.000083, 'reg': 0.00357, 'batch_size': 256, 'epochs': 15}

    loader_train, loader_val, loader_test = make_loaders(
        X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, solver['batch_size'], seed
    )

    torch.manual_seed(seed)
    model = Res_PSM(
        num_predictors=len(PREDICTOR_KEYS),
        hidden_sizes=arch['hidden_sizes'], layernorms=arch['layernorms'], silu=arch['silu'],
        motor_params=motor_params, y_mean=y_scaler.mean_, y_std=y_scaler.scale_,
        lambda_prior=arch['lambda_prior'],
    )
    nn_weight_params = [p for n, p in model.residual_net.named_parameters() if 'bias' not in n]
    nn_bias_params = [p for n, p in model.residual_net.named_parameters() if 'bias' in n]
    psm_params = list(model.psm.parameters())
    optimizer = torch.optim.AdamW([
        {'params': nn_weight_params, 'weight_decay': solver['reg']},
        {'params': nn_bias_params, 'weight_decay': 0.0},
        {'params': psm_params, 'weight_decay': 0.0},
    ], lr=solver['lr'])

    model_path = (
        Path(__file__).parent / "regressors" / "weights"
        / f"PHY_RES{arch['hidden_sizes']}l{arch['layernorms']}_s{arch['silu']}_lp{arch['lambda_prior']:g}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}_bs{solver['batch_size']}_ep{solver['epochs']}"
        f"_fold{fold_idx}.pth"
    )

    model, train_loss_history, val_loss_history, _ = get_or_train_model(
        model_path=model_path, model=model, optimizer=optimizer,
        device=device, dtype=dtype,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_test,
        epochs=solver['epochs'], stats_every=100, save=True,
    )
    if train_loss_history is not None:
        print(f"  [res]  fold {fold_idx} final loss — train: {train_loss_history[-1]:.4f}, "
              f"val: {val_loss_history[-1]:.4f}")

    return get_predictions_mlp(model, device, dtype, {'test': X_test_t}, y_scaler)['test']


def run_grey(fold_idx, X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t,
             y_scaler, motor_params, device, dtype, seed):
    arch = {'hidden': 24, 'r_scale': 1, 'psi_scale': 1, 'ld_scale': 0.15, 'lq_scale': 0.15,
            'prior_reg': 1e-2, 'var_reg': 0}
    solver = {'lr': 1e-3, 'reg': 0.00357, 'batch_size': 256, 'epochs': 15}

    loader_train, loader_val, loader_test = make_loaders(
        X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, solver['batch_size'], seed
    )

    torch.manual_seed(seed)
    model = GreyBoxPMSM(
        n_predictors=len(PREDICTOR_KEYS),
        motor_params=motor_params, y_mean=y_scaler.mean_, y_std=y_scaler.scale_,
        hidden=arch['hidden'],
        r_scale=arch['r_scale'], psi_scale=arch['psi_scale'], ld_scale=arch['ld_scale'], lq_scale=arch['lq_scale'],
        prior_reg=arch['prior_reg'], var_reg=arch['var_reg'],
    )
    nn_weight_params = [p for n, p in model.modnet.named_parameters() if 'bias' not in n]
    nn_bias_params = [p for n, p in model.modnet.named_parameters() if 'bias' in n]
    psm_params = [model.log_R0, model.Psi0, model.log_Ld0, model.log_Lq0]
    optimizer = torch.optim.AdamW([
        {'params': nn_weight_params, 'weight_decay': solver['reg']},
        {'params': nn_bias_params, 'weight_decay': 0.0},
        {'params': psm_params, 'weight_decay': 0.0},
    ], lr=solver['lr'])

    model_path = (
        Path(__file__).parent / "regressors" / "weights"
        / f"GREY{arch['hidden']}_pr{arch['prior_reg']:g}_vr{arch['var_reg']:g}"
        f"_rs{arch['r_scale']:g}_ps{arch['psi_scale']:g}_lds{arch['ld_scale']}_lqs{arch['lq_scale']:g}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}_bs{solver['batch_size']}_ep{solver['epochs']}"
        f"_fold{fold_idx}.pth"
    )

    model, train_loss_history, val_loss_history, _ = get_or_train_model(
        model_path=model_path, model=model, optimizer=optimizer,
        device=device, dtype=dtype,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_test,
        epochs=solver['epochs'], stats_every=100, save=True,
    )
    if train_loss_history is not None:
        print(f"  [grey] fold {fold_idx} final loss — train: {train_loss_history[-1]:.4f}, "
              f"val: {val_loss_history[-1]:.4f}")

    return get_predictions_mlp(model, device, dtype, {'test': X_test_t}, y_scaler)['test']


def run_fold(fold_idx, fold, np_raw_all, device, dtype, seed):
    """Prepare one fold's data (fit scalers on this fold's train split only, to avoid
    leakage) and fit/evaluate all 4 models on it. Returns a dict of test-set predictions
    keyed by model name ('bil', 'mlp', 'res', 'grey'), plus the fold's ground truth."""
    np_raw = {
        'train': {k: v[fold['train_idx']] for k, v in np_raw_all.items()},
        'val': {k: v[fold['val_idx']] for k, v in np_raw_all.items()},
        'test': {k: v[fold['test_idx']] for k, v in np_raw_all.items()},
    }

    X_phys = {s: to_matrix(np_raw[s], PREDICTOR_KEYS) for s in ('train', 'val', 'test')}
    Y_phys = {s: to_matrix(np_raw[s], RESPONSE_KEYS) for s in ('train', 'val', 'test')}

    # Scalers are fit on this fold's train split only.
    x_scaler = StandardScaler().fit(X_phys['train'])
    y_scaler = StandardScaler().fit(Y_phys['train'])
    X_scaled = {s: x_scaler.transform(X_phys[s]) for s in ('train', 'val', 'test')}
    Y_scaled = {s: y_scaler.transform(Y_phys[s]) for s in ('train', 'val', 'test')}

    X_t = {s: torch.tensor(X_scaled[s], dtype=dtype) for s in ('train', 'val', 'test')}
    Y_t = {s: torch.tensor(Y_scaled[s], dtype=dtype) for s in ('train', 'val', 'test')}
    # "mixed" [scaled | phys] inputs for the physics-informed models: the PSM equations need
    # real physical units, but the residual/modulation net trains better on scaled inputs.
    X_mixed_t = {s: torch.tensor(np.concatenate([X_scaled[s], X_phys[s]], axis=1), dtype=dtype)
                 for s in ('train', 'val', 'test')}

    pred = {}
    pred['bil'], motor_params = run_bilinear(np_raw['train'], np_raw['test'])
    pred['mlp'] = run_mlp(
        fold_idx, X_t['train'], Y_t['train'], X_t['val'], Y_t['val'], X_t['test'], Y_t['test'],
        y_scaler, device, dtype, seed,
    )
    pred['res'] = run_res(
        fold_idx, X_mixed_t['train'], Y_t['train'], X_mixed_t['val'], Y_t['val'], X_mixed_t['test'], Y_t['test'],
        y_scaler, motor_params, device, dtype, seed,
    )
    pred['grey'] = run_grey(
        fold_idx, X_mixed_t['train'], Y_t['train'], X_mixed_t['val'], Y_t['val'], X_mixed_t['test'], Y_t['test'],
        y_scaler, motor_params, device, dtype, seed,
    )

    return pred, Y_phys['test']


if __name__ == "__main__":
    print('--------------------------------------------------------------------------------------')
    print('Prepare data from measurements')
    np_raw_all, n = import_measurements(SIGNAL_NAMES, DATASET_DIR, plot=False)

    print('--------------------------------------------------------------------------------------')
    print("Datasheet reference (fixed parameters, no fitting -> not cross validated): "
          "evaluate on the full dataset")
    X_all_be, _ = build_basis_expansion(np_raw_all)
    Y_all_phys = to_matrix(np_raw_all, RESPONSE_KEYS)
    pred_datasheet = get_predictions_bil(DATASHEET, {'all': X_all_be})['all']
    report_regression_metrics(pred_datasheet, Y_all_phys, RESPONSE_KEYS, split_name="Full dataset (datasheet)")
    print(f"Datasheet R={DATASHEET[0]:.5f} ohm, Ld={DATASHEET[1]:.6f} H, "
          f"Lq={DATASHEET[2]:.6f} H, Psi={DATASHEET[3]:.5f} Wb")

    print('--------------------------------------------------------------------------------------')
    print(f'{N_FOLDS}-fold cross validation: bilinear (OLS), MLP, physics+residual, greybox')
    fs = 1e5  # sample rate, Hz
    chunk_size = fs * 0.1  # 0.1-second chunks; pick something >> autocorrelation time
    folds = chunked_kfold_indices(n, chunk_size, n_folds=N_FOLDS, val_frac=VAL_FRAC)

    model_names = ['bil', 'mlp', 'res', 'grey']
    cv_results = {name: {key: {'rmse': [], 'r2': []} for key in RESPONSE_KEYS} for name in model_names}

    # Collected across folds so every row gets exactly one out-of-fold prediction (each chunk
    # is a test chunk in exactly one fold), letting us plot each model's CV predictions against
    # the whole time series afterwards.
    oof_idx_parts, oof_true_parts = [], []
    oof_pred_parts = {name: [] for name in model_names}

    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS}  "
              f"(train {len(fold['train_idx'])} / val {len(fold['val_idx'])} / test {len(fold['test_idx'])} rows) ---")
        fold_pred, Y_test_phys = run_fold(fold_idx, fold, np_raw_all, device, DTYPE, SEED)

        oof_idx_parts.append(fold['test_idx'])
        oof_true_parts.append(Y_test_phys)
        for name in model_names:
            metrics = report_regression_metrics(
                fold_pred[name], Y_test_phys, RESPONSE_KEYS, split_name=f"Fold {fold_idx + 1} test ({name})"
            )
            record_fold_metrics(cv_results, name, metrics)
            oof_pred_parts[name].append(fold_pred[name])

    print('--------------------------------------------------------------------------------------')
    print_cv_summary(cv_results, RESPONSE_KEYS, N_FOLDS)

    # Out-of-fold predictions vs ground truth, one plot per model, covering the whole dataset
    # (every point was predicted by a model that never saw it during training).
    oof_idx = np.concatenate(oof_idx_parts)
    oof_true = np.concatenate(oof_true_parts, axis=0)
    for name in model_names:
        oof_pred = np.concatenate(oof_pred_parts[name], axis=0)
        plot_predictions(
            {'test': oof_pred}, {'test': oof_true}, {'test': oof_idx}, RESPONSE_KEYS,
            title=f'{name} — out-of-fold CV predictions',
        )

    plt.show()
    plt.close('all')

# Regression of Id, Iq, om on Ud, Uq. See Readme.md
# ok: validation loss is roughly tracking training loss downward without beginning to diverge.
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Fractions of the total dataset held out for validation / testing: train_frac = 1 - VAL_FRAC - TEST_FRAC.
# Val is used during development (checkpoint bookkeeping, loss curves, the greybox model's saturation histogram).
# Test is never touched for model selection and is only used to report a final metric for every model at the end of the script.
VAL_FRAC = 0.15
TEST_FRAC = 0.15

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


def chunked_split(n_samples, chunk_size, val_frac=0.2, test_frac=0.2, seed=2):
    """Split n_samples contiguous rows into non-overlapping chunks, then assign whole chunks to train/val/test splits 
    to avoid leakage between neighboring samples assumed to be autocorrelated.

    val_frac and test_frac are each a fraction of the total number of chunks (not of what's left after removing the other split), 
    so e.g. val_frac=0.2, test_frac=0.2 gives a 60/20/20 train/val/test split.
    Either can be 0 to fall back to a two-way split.
    """
    logger.info('*** split into train, val and test splits ***')
    if not (0 <= val_frac < 1) or not (0 <= test_frac < 1) or val_frac + test_frac >= 1:
        raise ValueError(
            f"val_frac ({val_frac}) and test_frac ({test_frac}) must each be in [0, 1) "
            f"and sum to less than 1."
        )

    chunk_size = int(chunk_size)
    n_chunks = n_samples // chunk_size

    # Any samples beyond n_chunks * chunk_size don't fill a full chunk and
    # are dropped so every chunk has equal length. Log how many are lost.
    discarded = n_samples - n_chunks * chunk_size
    print(f"  discarding {discarded} / {n_samples} trailing samples "
          f"({n_chunks} full chunks of size {chunk_size})")

    chunk_ids = np.arange(n_chunks)

    # Peel off the test chunks first, then split what's left into train/val.
    # val_frac is expressed relative to the *total*, so rescale it relative
    # to the remaining (1 - test_frac) chunks for the second split.
    if test_frac > 0:
        trainval_chunks, test_chunks = train_test_split(
            chunk_ids, test_size=test_frac, random_state=seed
        )
    else:
        trainval_chunks, test_chunks = chunk_ids, np.array([], dtype=int)

    if val_frac > 0:
        relative_val_frac = val_frac / (1 - test_frac)
        train_chunks, val_chunks = train_test_split(
            trainval_chunks, test_size=relative_val_frac, random_state=seed
        )
    else:
        train_chunks, val_chunks = trainval_chunks, np.array([], dtype=int)

    def chunk_to_rows(chunks):
        if len(chunks) == 0:
            return np.array([], dtype=np.int64)
        idx = np.concatenate([
            np.arange(c * chunk_size, (c + 1) * chunk_size) for c in chunks
        ]).astype(np.int64)
        return np.sort(idx)

    train_idx = chunk_to_rows(train_chunks)
    val_idx = chunk_to_rows(val_chunks)
    test_idx = chunk_to_rows(test_chunks)
    return train_idx, val_idx, test_idx, train_chunks, val_chunks, test_chunks


def summarize_split(
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    loader_train, loader_val, loader_test,
    x_scaler, y_scaler,
    predictor_keys, response_keys,
    train_idx=None, val_idx=None, test_idx=None,
    train_chunks=None, val_chunks=None, test_chunks=None,
    n_preview=5,
):
    """Print shapes, scaler stats, and a data preview for a train/val/test split."""
    n_total = len(X_train) + len(X_val) + len(X_test)

    print("*** Split summary ***")

    print(f"\nTotal samples: {n_total}")
    print(f"Train samples: {len(X_train)} ({len(X_train) / n_total:.1%})")
    print(f"Val samples:   {len(X_val)} ({len(X_val) / n_total:.1%})")
    print(f"Test samples:  {len(X_test)} ({len(X_test) / n_total:.1%})")

    if train_chunks is not None and val_chunks is not None and test_chunks is not None:
        print(f"\nTrain chunks: {len(train_chunks)}")
        print(f"Val chunks:   {len(val_chunks)}")
        print(f"Test chunks:  {len(test_chunks)}")

    print(f"\nX_train shape: {tuple(X_train.shape)}  (predictors: {predictor_keys})")
    print(f"Y_train shape: {tuple(Y_train.shape)}  (responses:  {response_keys})")
    print(f"X_val shape:   {tuple(X_val.shape)}")
    print(f"Y_val shape:   {tuple(Y_val.shape)}")
    print(f"X_test shape:  {tuple(X_test.shape)}")
    print(f"Y_test shape:  {tuple(Y_test.shape)}")

    print(f"\nX scaler mean: {x_scaler.mean_}")
    print(f"X scaler std:  {x_scaler.scale_}")
    print(f"Y scaler mean: {y_scaler.mean_}")
    print(f"Y scaler std:  {y_scaler.scale_}")

    print(f"\nFirst {n_preview} rows of X_train (scaled):\n{X_train[:n_preview]}")
    print(f"\nFirst {n_preview} rows of Y_train (scaled):\n{Y_train[:n_preview]}")
    print(f"\nFirst {n_preview} rows of X_val (scaled):\n{X_val[:n_preview]}")
    print(f"\nFirst {n_preview} rows of Y_val (scaled):\n{Y_val[:n_preview]}")
    print(f"\nFirst {n_preview} rows of X_test (scaled):\n{X_test[:n_preview]}")
    print(f"\nFirst {n_preview} rows of Y_test (scaled):\n{Y_test[:n_preview]}")

    print(f"\nloader_train: {len(loader_train)} batches of size {loader_train.batch_size}")
    print(f"loader_val:   {len(loader_val)} batches of size {loader_val.batch_size}")
    print(f"loader_test:  {len(loader_test)} batches of size {loader_test.batch_size}")

    xb, yb = next(iter(loader_train))
    print(f"\nExample train batch -> X: {tuple(xb.shape)}, Y: {tuple(yb.shape)}")
    print(f"X batch sample:\n{xb[:3]}")
    print(f"Y batch sample:\n{yb[:3]}")


def to_matrix(d, keys):
    return np.stack([d[k] for k in keys], axis=1)  # N x P


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


def report_regression_metrics(pred, true, response_keys, split_name=""):
    """Print RMSE and R^2 per response variable, in physical units."""
    print(f"*** {split_name} metrics (physical units) ***")
    for i, key in enumerate(response_keys):
        err = pred[:, i] - true[:, i]
        rmse = np.sqrt(np.mean(err ** 2))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  {key}: RMSE = {rmse:.4f}  |  R^2 = {r2:.4f}")


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


if __name__ == "__main__":
    # prepare data
    print('--------------------------------------------------------------------------------------')
    print('Prepare data from measurements')
    # --- 1. Load raw measurements ---
    np_raw_all, n = import_measurements(SIGNAL_NAMES, DATASET_DIR, plot=False)

    # --- 2. Chunked train/val/test split, to avoid leakage ---
    fs = 1e5  # sample rate, Hz
    chunk_size = fs * 0.1  # 0.1-second chunks; pick something >> autocorrelation time
    train_idx, val_idx, test_idx, train_chunks, val_chunks, test_chunks = chunked_split(
        n, chunk_size, val_frac=VAL_FRAC, test_frac=TEST_FRAC
    )
    np_raw_train = {k: v[train_idx] for k, v in np_raw_all.items()}
    np_raw_val = {k: v[val_idx] for k, v in np_raw_all.items()}
    np_raw_test = {k: v[test_idx] for k, v in np_raw_all.items()}

    X_train_phys = to_matrix(np_raw_train, PREDICTOR_KEYS)
    X_val_phys = to_matrix(np_raw_val, PREDICTOR_KEYS)
    X_test_phys = to_matrix(np_raw_test, PREDICTOR_KEYS)
    Y_train_phys = to_matrix(np_raw_train, RESPONSE_KEYS)
    Y_val_phys = to_matrix(np_raw_val, RESPONSE_KEYS)
    Y_test_phys = to_matrix(np_raw_test, RESPONSE_KEYS)

    # --- 3. Normalize (val/test splits: on train stats) ---
    x_scaler = StandardScaler().fit(X_train_phys)
    y_scaler = StandardScaler().fit(Y_train_phys)

    X_train_scaled = x_scaler.transform(X_train_phys)
    X_val_scaled = x_scaler.transform(X_val_phys)
    X_test_scaled = x_scaler.transform(X_test_phys)
    Y_train_scaled = y_scaler.transform(Y_train_phys)
    Y_val_scaled = y_scaler.transform(Y_val_phys)
    Y_test_scaled = y_scaler.transform(Y_test_phys)

    # Ground truth + row-index dicts reused by every plot_predictions() call below.
    Y_phys = {'train': Y_train_phys, 'val': Y_val_phys, 'test': Y_test_phys}
    idxs = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    print('--------------------------------------------------------------------------------------')
    print('As a baseline, use the bilinear physical model with the datasheet values')
    # --- 1. Model: Build basis expansion ---
    X_train_be, y_train_be = build_basis_expansion(np_raw_train)
    X_val_be, y_val_be = build_basis_expansion(np_raw_val)
    X_test_be, y_test_be = build_basis_expansion(np_raw_test)
    X_be = {'train': X_train_be, 'val': X_val_be, 'test': X_test_be}

    # --- 2: Compute and show baseline with datasheet values ---
    pred_bil = get_predictions_bil(DATASHEET, X_be)

    results_bil = plot_predictions(
        pred_bil, Y_phys, idxs, RESPONSE_KEYS, title='Physics model - baseline from datasheet'
    )

    for split in ('train', 'val', 'test'):
        report_regression_metrics(
            pred_bil[split], Y_phys[split], RESPONSE_KEYS, split_name=f"{split.capitalize()} (datasheet)"
        )
    print(f"Datasheet R={DATASHEET[0]:.5f} ohm, "
          f"Ld={DATASHEET[1]:.6f} H, "
          f"Lq={DATASHEET[2]:.6f} H, "
          f"Psi={DATASHEET[3]:.5f} Wb")

    print('--------------------------------------------------------------------------------------')
    print('Regress using the bilinear physical model and OLS')
    # --- 1. Compute the OLS solution ---
    betahat_bil = np.linalg.pinv(X_train_be) @ y_train_be
    motor_params_bil = dict(zip(MOTOR_PARAMETERS, betahat_bil.tolist()))

    # --- 2. Get predictions on all splits, including the held-out test set ---
    pred_bil = get_predictions_bil(betahat_bil, X_be)

    results_bil = plot_predictions(
        pred_bil, Y_phys, idxs, RESPONSE_KEYS, title='Physics model'
    )

    for split in ('train', 'val', 'test'):
        report_regression_metrics(
            pred_bil[split], Y_phys[split], RESPONSE_KEYS, split_name=f"{split.capitalize()} (bil)"
        )
    print(f"Learned R={motor_params_bil['R']:.5f} ohm, "
          f"Ld={motor_params_bil['Ld']:.6f} H, "
          f"Lq={motor_params_bil['Lq']:.6f} H, "
          f"Psi={motor_params_bil['Psi']:.5f} Wb")

    print('--------------------------------------------------------------------------------------')
    print('Regress using a multi-layer perceptron and a pytorch solver')
    # --- 1. Cast to tensor and build TensorDatasets ---
    X_train_scaled_t = torch.tensor(X_train_scaled, dtype=DTYPE)
    Y_train_scaled_t = torch.tensor(Y_train_scaled, dtype=DTYPE)
    X_val_scaled_t = torch.tensor(X_val_scaled, dtype=DTYPE)
    Y_val_scaled_t = torch.tensor(Y_val_scaled, dtype=DTYPE)
    X_test_scaled_t = torch.tensor(X_test_scaled, dtype=DTYPE)
    Y_test_scaled_t = torch.tensor(Y_test_scaled, dtype=DTYPE)
    train_ds = TensorDataset(X_train_scaled_t, Y_train_scaled_t)
    val_ds = TensorDataset(X_val_scaled_t, Y_val_scaled_t)
    test_ds = TensorDataset(X_test_scaled_t, Y_test_scaled_t)

    # --- 2. Define model architecture and solver parameters ---
    arch = {'hidden_sizes': [32, 32], 'layernorms': False, 'silu': False}
    solver = {'lr': 0.00043, 'reg': 0.00357, 'batch_size': 64, 'epochs': 5}

    loader_train = DataLoader(train_ds, batch_size=solver['batch_size'], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    loader_val = DataLoader(val_ds, batch_size=solver['batch_size'], shuffle=False)
    loader_test = DataLoader(test_ds, batch_size=solver['batch_size'], shuffle=False)

    # helper to debug DataLoaders if needed
    if logger.isEnabledFor(logging.INFO):
        debug_loader = DataLoader(train_ds, batch_size=solver['batch_size'], shuffle=False)  # no RNG touched
        summarize_split(
            X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test_scaled, Y_test_scaled,
            debug_loader, loader_val, loader_test,
            x_scaler, y_scaler,
            predictor_keys=PREDICTOR_KEYS,
            response_keys=RESPONSE_KEYS,
            train_chunks=train_chunks,
            val_chunks=val_chunks,
            test_chunks=test_chunks,
        )

    torch.manual_seed(SEED)  # identical model init regardless of what ran (or was skipped via a cached checkpoint) earlier in the script
    model = MLP(len(PREDICTOR_KEYS), arch['hidden_sizes'], len(RESPONSE_KEYS), layernorms=arch['layernorms'], silu=arch['silu'])
    optimizer = torch.optim.AdamW([  # AdamW if weight decays
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': solver['reg']},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0},
    ], lr=solver['lr'])

    MODEL_PATH = (
        Path(__file__).parent / "regressors" / "weights"
        / f"MLP{arch['hidden_sizes']}l{arch['layernorms']}_s{arch['silu']}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}"
        f"_bs{solver['batch_size']}_ep{solver['epochs']}.pth"
    )

    # --- 3. Train  ---
    model, train_loss_history, val_loss_history, test_loss = get_or_train_model(
        model_path=MODEL_PATH,
        model=model, optimizer=optimizer,
        device=device, dtype=DTYPE,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_test,
        epochs=solver['epochs'], stats_every=10, save=True
    )

    if train_loss_history is not None:
        plot_losses(train_loss_history, val_loss_history, title='Multi-Layer Perceptron')
    else:
        logger.info("  Skipped plotting — no training history (model was loaded from checkpoint).")

    # --- 4. Get predictions and compare them to measurements, on all three splits ---
    pred_mlp = get_predictions_mlp(
        model, device, DTYPE,
        {'train': X_train_scaled_t, 'val': X_val_scaled_t, 'test': X_test_scaled_t},
        y_scaler,
    )

    results_mlp = plot_predictions(
        pred_mlp, Y_phys, idxs, RESPONSE_KEYS, title='Multi-Layer Perceptron'
    )

    for split in ('train', 'val', 'test'):
        report_regression_metrics(
            pred_mlp[split], Y_phys[split], RESPONSE_KEYS, split_name=f"{split.capitalize()} (mlp)"
        )

    print('--------------------------------------------------------------------------------------')
    print('Regress using a physics + MLP residual model and a pytorch solver')
    # --- 1. Build tensor Datasets ---
    # The PSM equations only make sense in real amps/rad-s⁻¹/volts. But the residual NN trains better on scaled inputs.
    # So the model needs to see both: concatenate [scaled | phys] columns -> (N, 2*P)
    X_train_mixed_t = torch.tensor(np.concatenate([X_train_scaled, X_train_phys], axis=1), dtype=DTYPE)
    X_val_mixed_t = torch.tensor(np.concatenate([X_val_scaled, X_val_phys], axis=1), dtype=DTYPE)
    X_test_mixed_t = torch.tensor(np.concatenate([X_test_scaled, X_test_phys], axis=1), dtype=DTYPE)

    train_ds = TensorDataset(X_train_mixed_t, Y_train_scaled_t)
    val_ds = TensorDataset(X_val_mixed_t, Y_val_scaled_t)
    test_ds = TensorDataset(X_test_mixed_t, Y_test_scaled_t)

    # --- 2. Define model architecture and solver parameters ---
    arch = {'hidden_sizes': [24, 24], 'lambda_prior': 0.8, 'layernorms': True, 'silu': False}
    solver = {'lr': 0.000083, 'reg': 0.00357, 'batch_size': 256, 'epochs': 15}
    motor_params = motor_params_bil

    loader_train = DataLoader(train_ds, batch_size=solver['batch_size'], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    loader_val = DataLoader(val_ds, batch_size=solver['batch_size'], shuffle=False)
    loader_test = DataLoader(test_ds, batch_size=solver['batch_size'], shuffle=False)

    torch.manual_seed(SEED)  # identical model init regardless of what ran (or was skipped via a cached checkpoint) earlier in the script
    model = Res_PSM(
        num_predictors=len(PREDICTOR_KEYS),
        hidden_sizes=arch['hidden_sizes'], layernorms=arch['layernorms'], silu=arch['silu'],
        motor_params=motor_params, y_mean=y_scaler.mean_, y_std=y_scaler.scale_,
        lambda_prior=arch['lambda_prior']
    )
    psm_params = list(model.psm.parameters())
    nn_weight_params = [p for n, p in model.residual_net.named_parameters() if 'bias' not in n]
    nn_bias_params = [p for n, p in model.residual_net.named_parameters() if 'bias' in n]

    optimizer = torch.optim.AdamW([
        {'params': nn_weight_params, 'weight_decay': solver['reg']},
        {'params': nn_bias_params, 'weight_decay': 0.0},
        {'params': psm_params, 'weight_decay': 0.0},  # same lr as base group
    ], lr=solver['lr'])

    MODEL_PATH = (
        Path(__file__).parent / "regressors" / "weights"
        / f"PHY_RES{arch['hidden_sizes']}l{arch['layernorms']}_s{arch['silu']}_lp{arch['lambda_prior']:g}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}"
        f"_bs{solver['batch_size']}_ep{solver['epochs']}.pth"
    )

    # --- 3. Train ---
    model, train_loss_history, val_loss_history, test_loss = get_or_train_model(
        model_path=MODEL_PATH,
        model=model, optimizer=optimizer,
        device=device, dtype=DTYPE,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_test,
        epochs=solver['epochs'], stats_every=100, save=True
    )

    if train_loss_history is not None:
        plot_losses(train_loss_history, val_loss_history, title='PSM + residual model')
    else:
        logger.info("  Skipped plotting — no training history (model was loaded from checkpoint).")

    # --- 4. Get predictions and compare them to measurements, on all three splits ---
    pred_res = get_predictions_mlp(
        model, device, DTYPE,
        {'train': X_train_mixed_t, 'val': X_val_mixed_t, 'test': X_test_mixed_t},
        y_scaler,
    )

    results_res = plot_predictions(
        pred_res, Y_phys, idxs, RESPONSE_KEYS, title='PSM + residual model'
    )

    for split in ('train', 'val', 'test'):
        report_regression_metrics(
            pred_res[split], Y_phys[split], RESPONSE_KEYS, split_name=f"{split.capitalize()} (res)"
        )
    print(f"Learned R={model.psm.R.item():.5f} ohm, "
          f"Ld={model.psm.Ld.item():.6f} H, "
          f"Lq={model.psm.Lq.item():.6f} H, "
          f"Psi={model.psm.Psi.item():.5f} Wb")

    print('--------------------------------------------------------------------------------------')
    print('Regress using a greybox physics model and a pytorch solver')
    # --- 1. Build tensor Datasets ---
    # [can reuse train_ds/val_ds/test_ds computed from the mixed scaled | phys] tensors for the residual model above.

    # --- 2. Define model architecture and solver parameters ---
    arch = {'hidden': 24, 'r_scale': 1, 'psi_scale': 1, 'ld_scale': 0.15, 'lq_scale': 0.15, 'prior_reg': 1e-2, 'var_reg': 0}
    solver = {'lr': 1e-3, 'reg': 0.00357, 'batch_size': 256, 'epochs': 15}
    motor_params = motor_params_bil
    # motor_params = {'R': 0.03395023719471839, 'Lq': 6.830554781188803e-05, 'Psi': 0.0037917131323094203, 'Ld': 7.021647501503991e-05}
    # motor_params = dict(zip(MOTOR_PARAMETERS, DATASHEET))

    loader_train = DataLoader(train_ds, batch_size=solver['batch_size'], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    loader_val = DataLoader(val_ds, batch_size=solver['batch_size'], shuffle=False)
    loader_test = DataLoader(test_ds, batch_size=solver['batch_size'], shuffle=False)

    torch.manual_seed(SEED)  # identical model init regardless of what ran (or was skipped via a cached checkpoint) earlier in the script
    model = GreyBoxPMSM(
        n_predictors=len(PREDICTOR_KEYS),
        motor_params=motor_params, y_mean=y_scaler.mean_, y_std=y_scaler.scale_,
        hidden=arch['hidden'],
        r_scale=arch['r_scale'], psi_scale=arch['psi_scale'], ld_scale=arch['ld_scale'], lq_scale=arch['lq_scale'],
        prior_reg=arch['prior_reg'], var_reg=arch['var_reg']
    )

    nn_weight_params = [p for n, p in model.modnet.named_parameters() if 'bias' not in n]
    nn_bias_params = [p for n, p in model.modnet.named_parameters() if 'bias' in n]
    psm_params = [model.log_R0, model.Psi0, model.log_Ld0, model.log_Lq0]
    optimizer = torch.optim.AdamW([
        {'params': nn_weight_params, 'weight_decay': solver['reg']},
        {'params': nn_bias_params, 'weight_decay': 0.0},
        {'params': psm_params, 'weight_decay': 0.0},
    ], lr=solver['lr'])

    MODEL_PATH = (
        Path(__file__).parent / "regressors" / "weights"
        / f"GREY{arch['hidden']}_pr{arch['prior_reg']:g}_vr{arch['var_reg']:g}"
        f"_rs{arch['r_scale']:g}_ps{arch['psi_scale']:g}_lds{arch['ld_scale']}_lqs{arch['lq_scale']:g}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}"
        f"_bs{solver['batch_size']}_ep{solver['epochs']}.pth"
    )

    # --- 3. Train ---
    model, train_loss_history, val_loss_history, test_loss = get_or_train_model(
        model_path=MODEL_PATH,
        model=model, optimizer=optimizer,
        device=device, dtype=DTYPE,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_test,
        epochs=solver['epochs'], stats_every=100, save=True
    )

    if train_loss_history is not None:
        plot_losses(train_loss_history, val_loss_history, title='Greybox')
    else:
        logger.info("  Skipped plotting — no training history (model was loaded from checkpoint).")

    # --- 4. Get predictions and compare them to measurements, on all three splits ---
    pred_grey = get_predictions_mlp(
        model, device, DTYPE,
        {'train': X_train_mixed_t, 'val': X_val_mixed_t, 'test': X_test_mixed_t},
        y_scaler,
    )

    results_grey = plot_predictions(
        pred_grey, Y_phys, idxs, RESPONSE_KEYS, title='Greybox model'
    )

    for split in ('train', 'val', 'test'):
        report_regression_metrics(
            pred_grey[split], Y_phys[split], RESPONSE_KEYS, split_name=f"{split.capitalize()} (grey)"
        )

    train_data = compute_effective_params(model, device, DTYPE, loader_train)
    val_data = compute_effective_params(model, device, DTYPE, loader_val)
    test_data = compute_effective_params(model, device, DTYPE, loader_test)
    nominal = {
        'R': torch.exp(model.log_R0).item(), 'Psi': model.Psi0.item(),
        'Ld': torch.exp(model.log_Ld0).item(), 'Lq': torch.exp(model.log_Lq0).item(),
    }
    plot_effective_params(train_data, val_data, test_data, nominal)

    plt.show()
    plt.close('all')

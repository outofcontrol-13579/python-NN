# Regression of Id, Iq, om on Ud, Uq. See Readme.md
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
from regressors.nets import MLP, train, Res_PSM
from regressors.LSwConstr import LSwConstr

plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)
# wait = input("Press Enter to continue.")

logging.basicConfig(
    level=logging.WARNING,  # set to logging.INFO to get info-level output, set to logging.WARNING to silence
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

USE_GPU = False
DTYPE = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
elif USE_GPU and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Torch: running on', device)
np.random.seed(0)
torch.manual_seed(0)

DATASET_DIR = Path(__file__).parent.parent / "dime12" / "datasets"

# Signal names must match CSV column headers exactly.
SIGNAL_NAMES = ["Id", "Iq", "Ud", "Uq", "Wel"]

PREDICTOR_KEYS = ["Id", "Iq", "Wel"]
RESPONSE_KEYS = ["Ud", "Uq"]
MOTOR_PARAMETERS = ['Rd', 'Lq',
                    'Rq', 'Psi', 'Ld',]
DATASHEET = np.array([30e-3, 50e-6,
                      30e-3, 4.2e-3, 50e-6])
BIL = np.array([0.033950237195, 0.000068305548,
                0.033950237193, 0.003791713132, 0.000070216475])


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
    """Load raw signals from CSV files into a dict."""
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


def chunked_split(n_samples, chunk_size, val_frac=0.2, seed=42):
    logger.info('*** split into train and val splits ***')
    chunk_size = int(chunk_size)
    n_chunks = n_samples // chunk_size

    # Any samples beyond n_chunks * chunk_size don't fill a full chunk and
    # are dropped so every chunk has equal length. Log how many are lost.
    discarded = n_samples - n_chunks * chunk_size
    print(f"  discarding {discarded} / {n_samples} trailing samples "
          f"({n_chunks} full chunks of size {chunk_size})")

    chunk_ids = np.arange(n_chunks)
    train_chunks, val_chunks = train_test_split(
        chunk_ids, test_size=val_frac, random_state=seed
    )

    def chunk_to_rows(chunks):
        idx = np.concatenate([
            np.arange(c * chunk_size, (c + 1) * chunk_size) for c in chunks
        ]).astype(np.int64)
        return np.sort(idx)
    train_idx = chunk_to_rows(train_chunks)
    val_idx = chunk_to_rows(val_chunks)
    return train_idx, val_idx, train_chunks, val_chunks


def summarize_split(
    X_train, Y_train, X_val, Y_val,
    loader_train, loader_val,
    x_scaler, y_scaler,
    predictor_keys, response_keys,
    train_idx=None, val_idx=None,
    train_chunks=None, val_chunks=None,
    n_preview=5,
):
    """Print shapes, scaler stats, and a data preview for a train/val split."""
    n_total = len(X_train) + len(X_val)

    print("*** Split summary ***")

    print(f"\nTotal samples: {n_total}")
    print(f"Train samples: {len(X_train)} ({len(X_train) / n_total:.1%})")
    print(f"Val samples:   {len(X_val)} ({len(X_val) / n_total:.1%})")

    if train_chunks is not None and val_chunks is not None:
        print(f"\nTrain chunks: {len(train_chunks)}")
        print(f"Val chunks:   {len(val_chunks)}")

    print(f"\nX_train shape: {tuple(X_train.shape)}  (predictors: {predictor_keys})")
    print(f"Y_train shape: {tuple(Y_train.shape)}  (responses:  {response_keys})")
    print(f"X_val shape:   {tuple(X_val.shape)}")
    print(f"Y_val shape:   {tuple(Y_val.shape)}")

    print(f"\nX scaler mean: {x_scaler.mean_}")
    print(f"X scaler std:  {x_scaler.scale_}")
    print(f"Y scaler mean: {y_scaler.mean_}")
    print(f"Y scaler std:  {y_scaler.scale_}")

    print(f"\nFirst {n_preview} rows of X_train (scaled):\n{X_train[:n_preview]}")
    print(f"\nFirst {n_preview} rows of Y_train (scaled):\n{Y_train[:n_preview]}")
    print(f"\nFirst {n_preview} rows of X_val (scaled):\n{X_val[:n_preview]}")
    print(f"\nFirst {n_preview} rows of Y_val (scaled):\n{Y_val[:n_preview]}")

    print(f"\nloader_train: {len(loader_train)} batches of size {loader_train.batch_size}")
    print(f"loader_val:   {len(loader_val)} batches of size {loader_val.batch_size}")

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
        print(f"  Loaded model (saved val loss: {checkpoint.get('val_loss', 'n/a')}).")
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
    """Print RMSE and R^2 per response variable, in physical (inverse-transformed) units."""
    print(f"*** {split_name} metrics (physical units) ***")
    for i, key in enumerate(response_keys):
        err = pred[:, i] - true[:, i]
        rmse = np.sqrt(np.mean(err ** 2))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  {key}: RMSE = {rmse:.4f}  |  R^2 = {r2:.4f}")


def get_predictions_mlp(model, device, dtype, X_train, X_val, y_scaler):
    """Run the model on train/val inputs and inverse-transform to physical units."""
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        pred_train = model(X_train.to(device=device, dtype=dtype)).cpu().numpy()
        pred_val = model(X_val.to(device=device, dtype=dtype)).cpu().numpy()

    pred_train_orig = y_scaler.inverse_transform(pred_train)
    pred_val_orig = y_scaler.inverse_transform(pred_val)

    return pred_train_orig, pred_val_orig


def plot_predictions(
    pred_train_orig, pred_val_orig,
    Y_train_orig, Y_val_orig,
    train_idx, val_idx,
    response_keys, title=""
):
    """Plot predictions against ground truth in time order."""
    n_responses = len(response_keys)
    fig, axes = plt.subplots(n_responses, 1, figsize=(14, 3.5 * n_responses), sharex=True)
    fig.suptitle(title)
    if n_responses == 1:
        axes = [axes]

    for i, (ax, key) in enumerate(zip(axes, response_keys)):
        # Ground truth: plot both splits together in time order
        all_idx = np.concatenate([train_idx, val_idx])
        all_true = np.concatenate([Y_train_orig[:, i], Y_val_orig[:, i]])
        order = np.argsort(all_idx)
        ax.plot(all_idx[order], all_true[order], color='black', lw=0.8,
                label='ground truth', alpha=0.6, zorder=1)

        # Predictions: scatter, colored by split
        ax.scatter(train_idx, pred_train_orig[:, i], s=4, color='tab:blue',
                   label='pred (train)', alpha=0.6, zorder=2)
        ax.scatter(val_idx, pred_val_orig[:, i], s=4, color='tab:orange',
                   label='pred (val)', alpha=0.6, zorder=2)

        ax.set_ylabel(key)
        ax.legend(loc='upper right', markerscale=2)
        ax.set_title(f'{key}: predictions vs ground truth')

    axes[-1].set_xlabel('sample index (time)')
    plt.tight_layout()

    return {
        'pred_train': pred_train_orig, 'pred_val': pred_val_orig,
        'true_train': Y_train_orig, 'true_val': Y_val_orig,
    }


def build_basis_expansion(np_raw, d_num=2, q_num=3):
    """Build the interleaved (Ud/Uq) basis-expansion design matrix for a data split.

    Returns X (predictors) and y (targets), interleaved row-wise as
    [Ud-row, Uq-row, Ud-row, Uq-row, ...].
    id_: to avoid shadowing the build-in "id".
    """
    id_, iq, om, ud, uq = (np_raw[signal] for signal in ['Id', 'Iq', 'Wel', 'Ud', 'Uq'])
    num_meas = len(id_)

    cols = [id_, -om * iq,
            iq, om, om * id_]
    BE = None
    for col in cols:
        BE = col if BE is None else np.c_[BE, col]

    Ud_X = BE.copy()
    Ud_X[:, d_num:] = np.zeros((num_meas, q_num))
    Uq_X = BE.copy()
    Uq_X[:, :d_num] = np.zeros((num_meas, d_num))

    X = np.empty((num_meas * 2, d_num + q_num))
    X[::2] = Ud_X
    X[1::2] = Uq_X

    y = np.empty(num_meas * 2)
    y[::2] = ud
    y[1::2] = uq

    return X, y


def get_predictions_bil(betahat_cvx, X_be):
    """Compute (Ud, Uq) predictions in physical units for train and val splits.

    betahat_cvx : fitted parameter vector from LSwConstr
    X_be : dict with keys 'train' and 'val', each the predictor matrix
           returned by build_basis_expansion (interleaved Ud/Uq rows)

    Returns pred_train_orig, pred_val_orig, each shape (n_samples, 2) -> [Ud, Uq].
    """
    def _predict(X_split):
        pred_interleaved = X_split @ betahat_cvx
        pred_ud = pred_interleaved[0::2]
        pred_uq = pred_interleaved[1::2]
        return np.stack([pred_ud, pred_uq], axis=1)

    pred_train_orig = _predict(X_be['train'])
    pred_val_orig = _predict(X_be['val'])
    return pred_train_orig, pred_val_orig


if __name__ == "__main__":
    # prepare data
    print('--------------------------------------------------------------------------------------')
    print('Prepare data from measurements')
    # --- 1. Load raw measurements ---
    np_raw_all, n = import_measurements(SIGNAL_NAMES, DATASET_DIR, plot=False)

    # --- 2. Chunked train/val split, to avoid leakage ---
    fs = 1e5  # sample rate, Hz
    chunk_size = fs * 0.1  # 0.1-second chunks; pick something >> autocorrelation time
    train_idx, val_idx, train_chunks, val_chunks = chunked_split(n, chunk_size, val_frac=0.2)
    np_raw_train = {k: v[train_idx] for k, v in np_raw_all.items()}
    np_raw_val = {k: v[val_idx] for k, v in np_raw_all.items()}

    X_train_phys = to_matrix(np_raw_train, PREDICTOR_KEYS)
    X_val_phys = to_matrix(np_raw_val, PREDICTOR_KEYS)
    Y_train_phys = to_matrix(np_raw_train, RESPONSE_KEYS)
    Y_val_phys = to_matrix(np_raw_val, RESPONSE_KEYS)

    # --- 3. Normalize (val split: on train stats)  ---
    x_scaler = StandardScaler().fit(X_train_phys)
    y_scaler = StandardScaler().fit(Y_train_phys)

    X_train_scaled = x_scaler.transform(X_train_phys)
    X_val_scaled = x_scaler.transform(X_val_phys)
    Y_train_scaled = y_scaler.transform(Y_train_phys)
    Y_val_scaled = y_scaler.transform(Y_val_phys)

    print('--------------------------------------------------------------------------------------')
    print('As a baseline, use the bilinear physical model with the datasheet values')
    # --- 1. Model: Build basis expansion ---
    d_num, q_num = (2, 3)
    X_train_be, y_train_be = build_basis_expansion(np_raw_train, d_num, q_num)
    X_val_be, y_val_be = build_basis_expansion(np_raw_val, d_num, q_num)

    # --- 2: Compute and show baseline with datasheet values ---
    pred_train_phys_bil, pred_val_phys_bil = get_predictions_bil(
        DATASHEET, {'train': X_train_be, 'val': X_val_be}
    )

    results_bil = plot_predictions(
        pred_train_phys_bil, pred_val_phys_bil,
        Y_train_phys, Y_val_phys,
        train_idx, val_idx,
        RESPONSE_KEYS, title='Physics model - baseline from datasheet'
    )

    report_regression_metrics(pred_train_phys_bil, Y_train_phys, RESPONSE_KEYS, split_name="Train (datasheet)")
    report_regression_metrics(pred_val_phys_bil, Y_val_phys, RESPONSE_KEYS, split_name="Val (datasheet)")
    print(f"Datasheet Rd={DATASHEET[0]:.5f} ohm, Rq={DATASHEET[2]:.5f} ohm, "
          f"Ld={DATASHEET[4]:.6f} H, "
          f"Lq={DATASHEET[1]:.6f} H, "
          f"Psi={DATASHEET[3]:.5f} Wb")

    print('--------------------------------------------------------------------------------------')
    print('Regress using the bilinear physical model (with Rd == Rq constraint) and a QP solver')
    # --- 1. Build constraint matrix and run QP ---
    C = np.zeros((1, d_num + q_num))
    C[0, 0], C[0, 2] = 1, -1  # enforce R_d == R_q

    df_bil, betahat_bil = LSwConstr(y_train_be, X_train_be, C, MOTOR_PARAMETERS, y_val_be, X_val_be)[:2]

    # --- 4. Get predictions and compare them to measurements ---
    pred_train_phys_bil, pred_val_phys_bil = get_predictions_bil(
        betahat_bil, {'train': X_train_be, 'val': X_val_be}
    )

    results_bil = plot_predictions(
        pred_train_phys_bil, pred_val_phys_bil,
        Y_train_phys, Y_val_phys,
        train_idx, val_idx,
        RESPONSE_KEYS, title='Physics model'
    )

    report_regression_metrics(pred_train_phys_bil, Y_train_phys, RESPONSE_KEYS, split_name="Train (bil)")
    report_regression_metrics(pred_val_phys_bil, Y_val_phys, RESPONSE_KEYS, split_name="Val (bil)")
    print(f"Learned Rd={df_bil['coef']['Rd']:.5f} ohm, Rq={df_bil['coef']['Rq']:.5f} ohm, "
          f"Ld={df_bil['coef']['Ld']:.6f} H, "
          f"Lq={df_bil['coef']['Lq']:.6f} H, "
          f"Psi={df_bil['coef']['Psi']:.5f} Wb")

    print('--------------------------------------------------------------------------------------')
    print('Regress using a multi-layer perceptron and a pytorch solver')
    # --- 1. Cast to tensor and build TensorDatasets ---
    X_train_scaled_t = torch.tensor(X_train_scaled, dtype=DTYPE)
    Y_train_scaled_t = torch.tensor(Y_train_scaled, dtype=DTYPE)
    X_val_scaled_t = torch.tensor(X_val_scaled, dtype=DTYPE)
    Y_val_scaled_t = torch.tensor(Y_val_scaled, dtype=DTYPE)
    train_ds = TensorDataset(X_train_scaled_t, Y_train_scaled_t)
    val_ds = TensorDataset(X_val_scaled_t, Y_val_scaled_t)

    # --- 2. Define model architecture and solver parameters ---
    arch = {'hidden_sizes': [32, 32], 'layernorms': False, 'silu': False}
    solver = {'lr': 0.00043, 'reg': 0.00357, 'batch_size': 64, 'epochs': 5}

    loader_train = DataLoader(train_ds, batch_size=solver['batch_size'], shuffle=True, drop_last=True)
    loader_val = DataLoader(val_ds, batch_size=solver['batch_size'], shuffle=False)

    # helper to debug DataLoaders if needed
    if logger.isEnabledFor(logging.INFO):
        summarize_split(
            X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled,
            loader_train, loader_val,
            x_scaler, y_scaler,
            predictor_keys=PREDICTOR_KEYS,
            response_keys=RESPONSE_KEYS,
            train_chunks=train_chunks,
            val_chunks=val_chunks,
        )

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

    # --- 3. Train ---
    # Note - l2_loss here is not comparable to the QP loss above, where there were in effect twice as many samples and they were not normalized.
    # For a meaningful comparaison, use the regression metrics instead.
    model, train_loss_history, val_loss_history, test_loss = get_or_train_model(
        model_path=MODEL_PATH,
        model=model, optimizer=optimizer,
        device=device, dtype=DTYPE,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_val,
        epochs=solver['epochs'], stats_every=10, save=True
    )

    if train_loss_history is not None:
        plot_losses(train_loss_history, val_loss_history, title='Multi-Layer Perceptron')
    else:
        logger.info("  Skipped plotting — no training history (model was loaded from checkpoint).")

    # --- 4. Get predictions and compare them to measurements ---
    pred_train_phys_mlp, pred_val_phys_mlp = get_predictions_mlp(
        model, device, DTYPE, X_train_scaled_t, X_val_scaled_t, y_scaler,
    )

    results_mlp = plot_predictions(
        pred_train_phys_mlp, pred_val_phys_mlp,
        Y_train_phys, Y_val_phys,
        train_idx, val_idx,
        RESPONSE_KEYS, title='Multi-Layer Perceptron'
    )

    report_regression_metrics(pred_train_phys_mlp, Y_train_phys, RESPONSE_KEYS, split_name="Train (mlp)")
    report_regression_metrics(pred_val_phys_mlp, Y_val_phys, RESPONSE_KEYS, split_name="Val (mlp)")

    print('--------------------------------------------------------------------------------------')
    print('Regress using a physics + MLP residual model and a pytorch solver')
    # --- 1. Build tensor Datasets ---
    # The PSM equations only make sense in real amps/rad-s⁻¹/volts. But the residual NN trains better on scaled inputs.
    # So the model needs to see both: concatenate [scaled | phys] columns -> (N, 2*P)
    X_train_mixed_t = torch.tensor(np.concatenate([X_train_scaled, X_train_phys], axis=1), dtype=DTYPE)
    X_val_mixed_t = torch.tensor(np.concatenate([X_val_scaled, X_val_phys], axis=1), dtype=DTYPE)

    train_ds = TensorDataset(X_train_mixed_t, Y_train_scaled_t)
    val_ds = TensorDataset(X_val_mixed_t, Y_val_scaled_t)

    # --- 2. Define model architecture and solver parameters ---
    arch = {'hidden_sizes': [24, 24], 'layernorms': True, 'silu': False}
    solver = {'lr': 0.000083, 'reg': 0.00357, 'batch_size': 256, 'epochs': 15}
    motor_params = df_bil["coef"].rename({"Rd": "R"}).drop("Rq").astype(float).to_dict()

    loader_train = DataLoader(train_ds, batch_size=solver['batch_size'], shuffle=True, drop_last=True)
    loader_val = DataLoader(val_ds, batch_size=solver['batch_size'], shuffle=False)

    model = Res_PSM(
        num_predictors=len(PREDICTOR_KEYS),
        hidden_sizes=arch['hidden_sizes'], layernorms=arch['layernorms'], silu=arch['silu'],
        motor_params=motor_params, y_mean=y_scaler.mean_, y_std=y_scaler.scale_,
        learn_psm=True
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
        / f"PHY_RES{arch['hidden_sizes']}l{arch['layernorms']}_s{arch['silu']}"
        f"_lr{solver['lr']:g}_reg{solver['reg']:g}"
        f"_bs{solver['batch_size']}_ep{solver['epochs']}.pth"
    )

    # --- 3. Train ---
    model, train_loss_history, val_loss_history, test_loss = get_or_train_model(
        model_path=MODEL_PATH,
        model=model, optimizer=optimizer,
        device=device, dtype=DTYPE,
        loader_train=loader_train, loader_val=loader_val, loader_test=loader_val,
        epochs=solver['epochs'], stats_every=100, save=True
    )

    if train_loss_history is not None:
        plot_losses(train_loss_history, val_loss_history, title='PSM + residual model')
    else:
        logger.info("  Skipped plotting — no training history (model was loaded from checkpoint).")

    # --- 4. Get predictions and compare them to measurements ---
    pred_train_phys_res, pred_val_phys_res = get_predictions_mlp(
        model, device, DTYPE, X_train_mixed_t, X_val_mixed_t, y_scaler,
    )

    results_res = plot_predictions(
        pred_train_phys_res, pred_val_phys_res,
        Y_train_phys, Y_val_phys,
        train_idx, val_idx,
        RESPONSE_KEYS, title='PSM + residual model'
    )

    report_regression_metrics(pred_train_phys_res, Y_train_phys, RESPONSE_KEYS, split_name="Train (res)")
    report_regression_metrics(pred_val_phys_res, Y_val_phys, RESPONSE_KEYS, split_name="Val (res)")
    print(f"Learned R={model.psm.R.item():.5f} ohm, "
          f"Ld={model.psm.Ld.item():.6f} H, "
          f"Lq={model.psm.Lq.item():.6f} H, "
          f"Psi={model.psm.Psi.item():.5f} Wb")

    plt.show()
    plt.close('all')

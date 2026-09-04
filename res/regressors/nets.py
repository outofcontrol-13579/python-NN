import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import matplotlib.pyplot as plt
import time


class ParamModNet(nn.Module):
    """
    Shared trunk + heads producing raw modulation logits for the PMSM parameters.
    Consumes only the SCALED predictor block of the input.

    A shared trunk lets the corrections draw on a common latent cause
    (e.g. temperature, which drives both R and Psi) while separate heads
    still let them respond independently -- e.g. Psi additionally reacting
    to id (cross-saturation) in a way R does not.

    The heads are zero-initialized so the model starts training at
    nominal values (R_eff = R0, Psi_eff = Psi0, etc...).
    """

    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_R = nn.Linear(hidden, 1)
        self.head_Psi = nn.Linear(hidden, 1)
        self.head_Ld = nn.Linear(hidden, 1)
        self.head_Lq = nn.Linear(hidden, 1)

        for head in (self.head_R, self.head_Psi, self.head_Ld, self.head_Lq):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x_scaled):
        h = self.trunk(x_scaled)
        return (self.head_R(h).squeeze(-1), self.head_Psi(h).squeeze(-1),
                self.head_Ld(h).squeeze(-1), self.head_Lq(h).squeeze(-1))


class GreyBoxPMSM(nn.Module):
    """
    motor_params: dict {"R": ..., "Ld": ..., "Lq": ..., "Psi": ...}
    r_scale: max fractional deviation allowed for R
    """

    def __init__(self, n_predictors, motor_params, y_mean, y_std, hidden=16,
                 r_scale=0.3, psi_scale=0.2, ld_scale=0.15, lq_scale=0.15, prior_reg=1e-2, var_reg=1e-2):
        super().__init__()

        self.n_predictors = n_predictors
        self.prior_reg = prior_reg
        self.var_reg = var_reg

        if motor_params is None:
            motor_params = {"R": 30e-3, "Ld": 50e-6, "Lq": 50e-6, "Psi": 4e-3}

        # log-parameterize R0, Ld0, Lq0 so they stays strictly positive under any update
        self.log_R0 = nn.Parameter(torch.log(torch.tensor(float(motor_params["R"]))))
        self.log_Ld0 = nn.Parameter(torch.log(torch.tensor(float(motor_params["Ld"]))))
        self.log_Lq0 = nn.Parameter(torch.log(torch.tensor(float(motor_params["Lq"]))))
        self.Psi0 = nn.Parameter(torch.tensor(float(motor_params["Psi"])))

        self.r_scale, self.psi_scale = r_scale, psi_scale
        self.ld_scale, self.lq_scale = ld_scale, lq_scale
        self.modnet = ParamModNet(in_dim=n_predictors, hidden=hidden)

        # Fixed (non-learnable) output rescaling, one pair per output (ud, uq)
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))

    def forward(self, x):
        """
        x: (N, 2*P) = [scaled predictors | raw physical predictors]
        returns: (N, 2) scaled (ud, uq) predictions
        """
        P = self.n_predictors
        x_scaled, x_raw = x[:, :P], x[:, P:]
        id_, iq_, om_ = x_raw[:, 0], x_raw[:, 1], x_raw[:, 2]

        raw_R, raw_Psi, raw_Ld, raw_Lq = self.modnet(x_scaled)

        R0, Ld0, Lq0 = torch.exp(self.log_R0), torch.exp(self.log_Ld0), torch.exp(self.log_Lq0)
        R_eff = R0 * (1.0 + self.r_scale * torch.tanh(raw_R))
        Psi_eff = self.Psi0 * (1.0 + self.psi_scale * torch.tanh(raw_Psi))
        Ld_eff = Ld0 * (1.0 + self.ld_scale * torch.tanh(raw_Ld))
        Lq_eff = Lq0 * (1.0 + self.lq_scale * torch.tanh(raw_Lq))

        # Pull effective params toward nominal unless data demands otherwise.
        # Important near om ~ 0, where Psi_eff, Ld_eff, Lq_eff are unidentifiable from data alone.
        eps = 1e-8
        deviations = (
            ((R_eff - R0) / (R0 + eps)) ** 2 +
            ((Psi_eff - self.Psi0) / (self.Psi0 + eps)) ** 2 +
            ((Ld_eff - Ld0) / (Ld0 + eps)) ** 2 +
            ((Lq_eff - Lq0) / (Lq0 + eps)) ** 2
        )  # shape (N,)
        self.prior_loss = deviations.mean()       # mean drift, comparable to PSM's prior
        self.variance_loss = deviations.var()     # penalizes spread of corrections

        # transient per-batch values
        self.R_eff, self.Psi_eff = R_eff.detach(), Psi_eff.detach()
        self.Ld_eff, self.Lq_eff = Ld_eff.detach(), Lq_eff.detach()

        ud = R_eff * id_ - om_ * Lq_eff * iq_
        uq = R_eff * iq_ + om_ * (Ld_eff * id_ + Psi_eff)

        u_phys = torch.stack([ud, uq], dim=1)
        return (u_phys - self.y_mean) / self.y_std


class PSM(nn.Module):
    """
    Steady-state PSM d/q voltage equations:
        ud = R*id - om*Lq*iq
        uq = R*iq + om*(Ld*id + Psi)

    R, Ld, Lq, Psi are learnable parameters, initialized from motor_params.

    Also computes self.prior_loss on every forward call: the squared relative
    deviation of R/Ld/Lq/Psi from their nominal (construction-time) values.
    Consumers weight this by their own regularization strength -- see
    Res_PSM.lambda_prior and train().
    """

    def __init__(self, R_init, Ld_init, Lq_init, Psi_init, y_mean, y_std):
        super().__init__()

        R_init_t = torch.tensor(R_init, dtype=torch.float32)
        Ld_init_t = torch.tensor(Ld_init, dtype=torch.float32)
        Lq_init_t = torch.tensor(Lq_init, dtype=torch.float32)
        Psi_init_t = torch.tensor(Psi_init, dtype=torch.float32)

        # Nominal values, frozen at construction time, for the prior-loss regularizer below.
        self.register_buffer('R0', R_init_t.clone())
        self.register_buffer('Ld0', Ld_init_t.clone())
        self.register_buffer('Lq0', Lq_init_t.clone())
        self.register_buffer('Psi0', Psi_init_t.clone())

        # Learnable parameters.
        self.R = nn.Parameter(R_init_t)
        self.Ld = nn.Parameter(Ld_init_t)
        self.Lq = nn.Parameter(Lq_init_t)
        self.Psi = nn.Parameter(Psi_init_t)

        # Fixed (non-learnable) rescaling constants, one pair per output (v_d, v_q)
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))  # shape (2,)
        self.register_buffer('y_std', torch.tensor(y_std, dtype=torch.float32))    # shape (2,)

    def forward(self, i_d, i_q, omega):
        eps = 1e-8
        self.prior_loss = (
            ((self.R - self.R0) / (self.R0 + eps)) ** 2 +
            ((self.Ld - self.Ld0) / (self.Ld0 + eps)) ** 2 +
            ((self.Lq - self.Lq0) / (self.Lq0 + eps)) ** 2 +
            ((self.Psi - self.Psi0) / (self.Psi0 + eps)) ** 2
        )

        u_d = self.R * i_d - omega * self.Lq * i_q
        u_q = self.R * i_q + omega * (self.Ld * i_d + self.Psi)
        u_psm_phys = torch.stack([u_d, u_q], dim=1)
        return (u_psm_phys - self.y_mean) / self.y_std


class Res_PSM(nn.Module):
    """
    u_pred = u_psm(i_d, i_q, omega; R, Ld, Lq, Psi) + NN(i_d, i_q, omega)

    Expects input x of shape (N, 2*P): first P columns are SCALED predictors
    (fed to the residual NN), last P columns are RAW physical-unit predictors
    (fed to the physics term), in the same column order.

    lambda_prior weights self.prior_loss (computed by the psm submodule on
    every forward call) in the training objective -- see train().
    """

    def __init__(self, num_predictors, hidden_sizes, layernorms=False, silu=False,
                 motor_params=None, y_mean=0, y_std=1, lambda_prior=0.8):
        super().__init__()

        if motor_params is None:
            motor_params = {"R": 30e-3, "Ld": 50e-6, "Lq": 50e-6, "Psi": 4e-3}

        self.num_predictors = num_predictors
        self.lambda_prior = lambda_prior
        self.psm = PSM(motor_params["R"], motor_params["Ld"], motor_params["Lq"], motor_params["Psi"],
                       y_mean, y_std)

        self.residual_net = MLP(num_predictors, hidden_sizes, 2, layernorms, silu)

        last_linear = self.residual_net.layers[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        P = self.num_predictors
        x_scaled, x_raw = x[:, :P], x[:, P:]
        i_d, i_q, omega = x_raw[:, 0], x_raw[:, 1], x_raw[:, 2]

        u_psm_scaled = self.psm(i_d, i_q, omega)
        u_res_scaled = self.residual_net(x_scaled)
        self.prior_loss = self.psm.prior_loss  # expose for train()'s regularizer
        return u_psm_scaled + u_res_scaled   # both terms in scaled units


class MLP(nn.Module):
    def __init__(self, num_predictors, hidden_sizes, num_responses, layernorms=False, silu=False):
        super().__init__()
        layers = []
        in_dim = num_predictors
        out_dim = num_responses
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h)]
            if layernorms:
                layers += [nn.LayerNorm(h)]
            layers += [nn.SiLU() if silu else nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def check_loss(device, dtype, loader, model):
    """Compute mean MSE loss over an entire DataLoader."""
    model.eval()
    total_sq_error = 0.0
    total_elements = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            preds = model(x)
            total_sq_error += F.mse_loss(preds, y, reduction='sum').item()
            total_elements += y.numel()
    return total_sq_error / total_elements


# def grad_norm_of(params):
#     """L2 norm of gradients for a given iterable of parameters, without clipping."""
#     norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
#     if not norms:
#         return torch.tensor(0.0)
#     return torch.norm(torch.stack(norms), 2)


def train(device, dtype, loader_train, loader_val, loader_test, model, optimizer, epochs=1, stats_every=100):
    """
    Returns: training and validation losses for each epoch and test loss computed with best validation parameters
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    batches_per_epoch = len(loader_train)

    def warmup_lambda(step):
        return min(1.0, step / batches_per_epoch)  # one epoch of warmup

    # Learning Rate Schedule is designed to be complete, with one epoch of warm-up followed by cosine annealing until the last epoch.
    # Therefore no point in saving optimizer state to resume training from a checkpoint.
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1), eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Storage containers
    val_losses = np.full(epochs, np.nan)
    train_losses = np.full(epochs, np.nan)
    losses, grad_norms, steps = [], [], []
    # lrs = []  # debug learning rate scheduler

    # PSM-parameter tracking (only if the model has a `psm` submodule or is an instance of GreyBoxPSM)
    has_psm = getattr(model, 'psm', None) is not None
    has_greypsm = isinstance(model, GreyBoxPMSM)
    if has_psm or has_greypsm:
        psm_param_history = {'R': [], 'Ld': [], 'Lq': [], 'Psi': []}
        psm_grad_norms, res_grad_norms = [], []
    else:
        psm_param_history = psm_grad_norms = res_grad_norms = None

    # Best-checkpoint tracking
    best_val_loss = float('inf')
    best_params = None  # will hold a deep copy of state_dict

    for e in range(epochs):
        tic = time.time()
        model.train()  # put model to training mode
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            preds = model(x)
            loss = F.mse_loss(preds, y)

            if has_psm:
                loss = loss + model.lambda_prior * model.prior_loss
            elif has_greypsm:
                loss = loss + model.prior_reg * model.prior_loss + model.var_reg * model.variance_loss
            # Compute the gradient of the loss with respect to each  parameter of the model.
            loss.backward()

            # Compute pre-clip gradient norm for monitoring, then clip to prevent the occasional exploding gradient
            if has_psm:
                res_grad_norm = torch.nn.utils.clip_grad_norm_(list(model.residual_net.parameters()), max_norm=5.0)
                psm_grad_norm = torch.nn.utils.clip_grad_norm_(list(model.psm.parameters()), max_norm=5e5)
                grad_norm = res_grad_norm + psm_grad_norm
            elif has_greypsm:
                psm_params = [model.log_R0, model.Psi0, model.log_Ld0, model.log_Lq0]
                res_grad_norm = torch.nn.utils.clip_grad_norm_(model.modnet.parameters(), max_norm=5.0)
                psm_grad_norm = torch.nn.utils.clip_grad_norm_(psm_params, max_norm=5e5)
                grad_norm = res_grad_norm + psm_grad_norm
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 1.0

            # Update the parameters of the model using the gradients computed by the backwards pass.
            optimizer.step()
            if e == 0:
                warmup_scheduler.step()  # warmup the learning rate every step during the first epoch
            # lrs.append(optimizer.param_groups[0]['lr'])  # debug learning rate scheduler

            if stats_every is not None and t % stats_every == 0:
                losses.append(loss.item())
                grad_norms.append(grad_norm.item())
                steps.append(e * batches_per_epoch + t)  # global step count
                if has_psm or has_greypsm:
                    psm_grad_norms.append(psm_grad_norm.item())
                    res_grad_norms.append(res_grad_norm.item())

        train_loss = check_loss(device, dtype, loader_train, model)
        val_loss = check_loss(device, dtype, loader_val, model)
        toc = time.time() - tic
        print("(Epoch %d / %d) %.2f seconds. train loss: %f; val_loss: %f"
              % (e + 1, epochs, toc, train_loss, val_loss))
        if has_psm:
            print(f"    R={model.psm.R.item():.5f} ohm, "
                  f"Ld={model.psm.Ld.item():.6f} H, "
                  f"Lq={model.psm.Lq.item():.6f} H, "
                  f"Psi={model.psm.Psi.item():.5f} Wb")
            psm_param_history['R'].append(model.psm.R.item())
            psm_param_history['Ld'].append(model.psm.Ld.item())
            psm_param_history['Lq'].append(model.psm.Lq.item())
            psm_param_history['Psi'].append(model.psm.Psi.item())

        if has_greypsm:
            print(f"    R0={torch.exp(model.log_R0).item():.5f} ohm, "
                  f"Ld0={torch.exp(model.log_Ld0).item():.6f} H, "
                  f"Lq0={torch.exp(model.log_Lq0).item():.6f} H, "
                  f"Psi0={model.Psi0.item():.5f} Wb")
            psm_param_history['R'].append(torch.exp(model.log_R0).item())
            psm_param_history['Ld'].append(torch.exp(model.log_Ld0).item())
            psm_param_history['Lq'].append(torch.exp(model.log_Lq0).item())
            psm_param_history['Psi'].append(model.Psi0.item())

        train_losses[e] = train_loss
        val_losses[e] = val_loss

        # Save a snapshot of the parameters whenever validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = copy.deepcopy(model.state_dict())
            print("  --> New best val loss: %.7f — checkpoint saved." % best_val_loss)

        if e > 0:
            cosine_scheduler.step()  # adjust the learning rate at the end of each epoch

    # Restore the best parameters found during training
    if best_params is not None:
        model.load_state_dict(best_params)
        print("Restored best model parameters (val loss: %.7f)." % best_val_loss)

    # Evaluate on the test set using the best parameters
    test_loss = check_loss(device, dtype, loader_test, model)
    print("Test loss (best val checkpoint): %.7f" % test_loss)

    # Print training statistics
    if stats_every is not None:
        plot_stats(losses, grad_norms, steps, epochs, batches_per_epoch)
        if has_psm or has_greypsm:
            plot_physics_stats(psm_param_history, psm_grad_norms, res_grad_norms, steps, epochs, batches_per_epoch)

    # fig, axes = plt.subplots(); axes.plot(lrs)  # debug learning rate scheduler

    return train_losses, val_losses, test_loss


def plot_stats(losses, grad_norms, steps, epochs, batches_per_epoch):
    num_plots = 2
    fig, axes = plt.subplots(num_plots, 1)
    i = 0
    # --- Loss ---
    axes[i].plot(steps, losses)
    axes[i].set_title("Loss")
    axes[i].set_xlabel("Step")
    axes[i].grid(linestyle='--', linewidth=0.5)
    axes[i].set_ylabel("MSE (L2) loss")

    # --- Gradient norm ---
    i += 1
    axes[i].plot(steps, grad_norms, color='tab:orange')
    axes[i].set_title("Gradient Norm (pre-clip)")
    axes[i].set_xlabel("Step")
    axes[i].set_ylabel("L2 norm")
    axes[i].set_yscale("log")
    # A healthy run typically shows the norm decreasing gradually;
    # a spike is a bad batch or an LR that's too high;
    # a collapse toward zero in early layers (visible if add per-layer norms) signals vanishing gradients.

    for ax in axes:
        for epoch_idx in range(epochs):
            ax.axvline(x=epoch_idx * batches_per_epoch, color='gray',
                       linestyle=':', linewidth=0.8, alpha=0.5)

    plt.subplots_adjust(right=0.75)
    plt.tight_layout()


def plot_physics_stats(physics_param_history, psm_grad_norms, res_grad_norms, steps, epochs, batches_per_epoch):
    fig, axes = plt.subplots(3, 1, figsize=(7.5 * 1.618, 7.5))

    # --- Physics parameter values, one point per epoch ---
    epoch_x = np.arange(1, len(physics_param_history['R']) + 1)
    for key in ['R', 'Ld', 'Lq', 'Psi']:
        axes[0].plot(epoch_x, physics_param_history[key], marker='o', label=key)
    axes[0].set_title("PSM parameters over training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value (SI units)")
    axes[0].grid(linestyle='--', linewidth=0.5)
    axes[0].legend()

    # --- PSM param group gradient norm, per logged step ---
    axes[1].plot(steps, psm_grad_norms, color='tab:green')
    axes[1].set_title("PSM parameters — gradient norm (unclipped)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("L2 norm")
    axes[1].set_yscale("log")

    # --- Resnet param group gradient norm, per logged step ---
    axes[2].plot(steps, res_grad_norms, color='tab:red')
    axes[2].set_title("Resnet parameters — gradient norm (unclipped)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("L2 norm")
    axes[2].set_yscale("log")

    for epoch_idx in range(epochs):
        axes[1].axvline(x=epoch_idx * batches_per_epoch, color='gray',
                        linestyle=':', linewidth=0.8, alpha=0.5)
        axes[2].axvline(x=epoch_idx * batches_per_epoch, color='gray',
                        linestyle=':', linewidth=0.8, alpha=0.5)

    plt.tight_layout()

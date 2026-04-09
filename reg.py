# Neuronales Netzwerk (NN) für Regressionsaufgabe: 2 layers (100, 50) können sinc darstellen
# Intuition zur "Kapazität": siehe https://abursuc.github.io/slides/polytechnique/14-01-deeper.html#7

import time
import sys

import numpy as np
import matplotlib.pyplot as plt
# from autograd elementwise_grad as egrad, numpy as np 

# ── Reproduzierbarkeit & Darstellung ──────────────────────────────────────────────────────────
rng = np.random.default_rng(1303)
np.set_printoptions(
    formatter={"float": "{:1.1e}".format},
    linewidth=sys.maxsize,
    threshold=sys.maxsize,
)
plt.rcParams["figure.figsize"] = (7.5 * 1.618, 7.5)

# ── Naturgesetz an der Basis der Beobachtungen ────────────────────────────────────
P = 2  # Dimensionalität


def nature(vec):
    # # quadratische Funktion: x.T @ A @ x
    # A = np.eye(2); A[0, 1] = -0.5; A[1, 0] = 0.5
    # return np.sum(np.multiply(vec, (A @ vec)), axis=0)
    # # x1 * x2
    # return np.multiply(vec[0], vec[1])
    # # Gaußsche Dichtefunktion (pdf)
    # mu = np.array([0, 0]); Sigma = np.array([[1, 0.3], [0.3, 0.4]]) # mu als (2, )
    # invS = np.linalg.inv(Sigma)
    # det = 1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(Sigma)))
    # xminusmu = vec - mu.reshape(-1, 1)
    # num = np.exp(-0.5 * np.sum(np.multiply(xminusmu, (invS @ xminusmu)), axis=0))
    # return num / det
    # # Polynom
    # # beta = np.array([0.165, -0.364, -0.419, 0.23])
    # beta = np.array([1, -2, 1.1, 1])
    # # X = np.array([vec[0], -vec[1], -vec[1]**3, -0.5*vec[0]**2*vec[1]]).T
    # X = np.array([vec[0]**5, vec[0]**3, vec[0], vec[0]*vec[1]**2]).T
    # return X @ beta
    # sinc
    return (np.sin(vec[0] * 10) / vec[0] / 10) * (np.sin(vec[1] * 10) / vec[1] / 10)


# ── Datensatz = beobachtetes X (Prädiktoren) und y (Zielvariable) ─────────────────────────────────
N = 1000  # Anzahl der Beobachtungen
max_x = 1
gran = max_x / 100
valsx = np.arange(-max_x, max_x, gran)
valsy = valsx
x_grid, y_grid = np.meshgrid(valsx, valsy)
valsall = np.array([x_grid.flatten(), y_grid.flatten()])
mask = rng.choice(np.arange(len(x_grid.flatten())), N)
X = np.c_[x_grid.flatten()[mask], y_grid.flatten()[mask]]   # N x P - jede Zeile = ein Datenpunkt
# X = X[np.linalg.norm(X, axis=1) > 0.35]                   # Intuition Generalisierung, teste 0.35 und 0.3
y = nature(X.T)                                             # N, - Beobachtungen (Rauschen hinzufügen: + rng.standard_normal(N)*0.1)
print(f"Dataset shapes — X: {X.shape}, y: {y.shape}")

# ── NN-Architektur & Hyperparameter ──────────────────────────────────────────────────
H1 = 100            # Neuronen in der ersten Schicht
H2 = 50             # Neuronen in der zweiten Schicht
LR = 0.2            # Lernrate (= Schrittweite des Gradientenabstiegs)
REG = 0.0           # L2-Regularisierungsstärke (auf > 0 setzen für Regularisierung)
MAX_ITER = 60_000
LOG_EVERY = 1_000   # Loss alle .. Iterationen ausgeben

# ── Modell (NN) und Verlustfunktion ────────────────────────────────────────────────────────────────
# gekapselt als Funktion - damit autograd als Alternative zum manuellen Gradienten verwendet werden kann
def f(vec, *, return_activations=False, predict=False, Xp=None, only_loss=True):
    """
    Gibt (Skalar loss, Gradientenvektor) zurück, berechnet in einem Vorwärts- und Rückwärtsdurchlauf.

    Modell:  z1 = h(X W1 + b1), mit h als Nichtlinearität wie z.B. ReLU (max(0, input)) oder tanh
             z2 = h(z1 W2 + b2)
             out = z2 Wout + bout
    Loss: L = 0.5 * ||out - y||² / N  +  0.5 * reg * (||W1||² + ||W2||² + ||Wout||²)
    """

    W1_, b1_, W2_, b2_, Wout_, bout_ = unpack(vec)

    # ── Vorwärtsdurchlauf ──
    # ── Standardmäßig Trainingsdaten X, kann auch zur Prädiktion genutzt werden ──
    if Xp is None:
        Xp = X

    # erste Schicht
    a1 = Xp @ W1_ + b1_         # N x H1   Voraktivierungen
    m1 = a1 > 0                 # N × H1   ReLU-Maske (cached für Rückwärtsdurchlauf)
    z1 = a1 * m1                # N × H1   Aktivierungen
    # z1 = np.tanh(a1)          # hier Aktivierung tauschen falls gewünscht

    # zweite Schicht
    a2 = z1 @ W2_ + b2_         # N x H2
    m2 = a2 > 0                 # N × H2   ReLU-Maske
    z2 = a2 * m2                # N × H2
    # z2 = np.tanh(a2)

    # Output Schicht
    out = z2 @ Wout_ + bout_    # N,

    if return_activations:      # Aktivierungshistogramme (nützlich zum Debuggen der Initialisierung)
        return [np.sum(z1, axis=0) / N, np.sum(z2, axis=0) / N]
    if predict:                 # nur Prädiktion zurückgeben (Vorwärtsdurchlauf), ohne loss/Gradientberechnung
        return out

    # ── loss ── (L2-loss + Regularisierung)
    residual = out - y          # N,
    dataloss = 0.5 * np.dot(residual, residual) / N
    regloss = 0.5 * REG * (
        np.dot(W1_.ravel(), W1_.ravel())
        + np.dot(W2_.ravel(), W2_.ravel())
        + np.dot(Wout_, Wout_)
    )
    loss = dataloss + regloss

    if only_loss:           # nur loss zurückgeben (dies ist die Schnittstelle für autograd)
        return loss
    
    # ── Rückwärtsdurchlauf: Gradienten der loss bzgl. Parameter berechnen (Backpropagation) ── 
    # Notation: d* = d(dataloss)/d*, siehe https://cs231n.github.io/optimization-2/#staged
    dout = residual / N         # N,

    # Output Schicht
    dWout = z2.T @ dout         # h2
    dbout = np.sum(dout)        # 1

    # zweite Schicht (ReLU: Null, wo Voraktivierung ≤ 0) 
    dz2 = np.outer(dout, Wout_) # N × H2
    dz2 *= m2                   # entspricht dz2[a2 <= 0] = 0, aber schneller
    dW2 = z1.T @ dz2            # H1 x H2
    db2 = np.sum(dz2, axis=0)   # H2,

    # erste Schicht
    dz1 = (dz2 @ W2_.T)         # N × H1
    dz1 *= m1       
    dW1 = Xp.T @ dz1            # P x H1
    db1 = np.sum(dz1, axis=0)   # H1,

    # Regularisierungsgradienten
    dW2 += REG * W2_
    dW1 += REG * W1_
    dWout += REG * Wout_

    dW = pack(dW1, db1, dW2, db2, dWout, dbout)
    return loss, dW


# ── Hilfsfunktionen zum Packen des Parametervektors ─────────────────────────────────────────────────
# Layout des Parametervektors W (alles zeilenweise abgeflacht und hintereinander gehängt):
#   W1 (P×H1) | b1 (H1) | W2 (H1×H2) | b2 (H2) | Wout (H2) | bout (1)


def pack(W1, b1, W2, b2, Wout, bout):
    return np.concatenate([a.ravel() for a in (W1, b1, W2, b2, Wout, bout)])


def unpack(vec):
    i = 0
    W1   = vec[i : i + P*H1].reshape(P, H1);         i += P*H1
    b1   = vec[i : i + H1];                          i += H1
    W2   = vec[i : i + H1*H2].reshape(H1, H2);       i += H1*H2
    b2   = vec[i : i + H2];                          i += H2
    Wout = vec[i : i + H2];                          i += H2  
    bout = vec[i : i + 1];                           i += 1
    return W1, b1, W2, b2, Wout, bout

# ── Initialisierung der Parameter ──────────────────────────────────────────────────────────
# siehe theoretische Begründung in https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
W1_init = rng.standard_normal((P, H1)) * np.sqrt(2.0 / (P + H1))
b1_init = np.zeros(H1)
W2_init = rng.standard_normal((H1, H2)) * np.sqrt(2.0 / (H1 + H2))
b2_init = np.zeros(H2)
Wout_init = rng.standard_normal(H2) * np.sqrt(2.0 / (H2 + 1))
bout_init = np.zeros(1)
W = pack(W1_init, b1_init, W2_init, b2_init, Wout_init, bout_init)

# ── gradient descent-Schleife ──────────────────────────────────────────────────────────────
losses = np.full(MAX_ITER, np.nan)
ratios = np.full(MAX_ITER, np.nan)
# # Autograd-Alternative für schnelle Modelländerungen (tanh, etc...): nächste Zeile und Import von autograd ganz oben ausführen
# g_f = egrad(f) 

tic = time.time()
for i in range(MAX_ITER):
    loss, dW = f(W, only_loss=False)
    # dW = g_f(W) # Autograd-Alternative: anstelle der vorherigen Zeile verwenden
    # print(np.allclose(dW_egrad, dW)) # Plausibilitätsprüfung: Autograd vs. manuell
    update = -LR * dW
    W += update

    # training loss und Größenverhältnis der updates tracken
    if i % 10 == 0:
        losses[i] = loss
        # losses[i] = f(W, only_loss=True) # Autograd-Alternative: anstelle der vorherigen Zeile verwenden
        ratios[i] = np.linalg.norm(update) / (np.linalg.norm(W) + 1e-12)  # Ziel ~1e-3

    if i % LOG_EVERY == 0:
        toc = time.time()
        print(f"iter {i:6d}  loss {losses[i]:.6e}  "
              f"update/param ratio {ratios[i]:.2e}  "
              f"learning rate {LR:.2e}  "
              f"({toc - tic:.1f}s)")
        tic = toc
        # # Zum Debuggen von Initialisierung und Lernrate: https://arxiv.org/pdf/1206.5533v2
        # # Bei ReLU sollten nicht zu viele Nullen (sog. „tote Neuronen“) in einer Schicht auftreten. Falls doch, ist die Lernrate zu hoch.
        # # Grund: Ein großer Gradient kann die Gewichte so verändern, dass das Neuron nie wieder aktiviert wird (=> Gradient bleibt dauerhaft null, ungünstig).
        # # Bei tanh sollten die Werte über den gesamten Bereich [-1, 1] verteilt sein, nicht nur bei 0 oder -1 oder 1.
        # activations = f(W, return_activations=True)
        # fig, ax = plt.subplots()
        # bottom_distance = 0.01
        # ax.hist(activations, bins=50, density=False, bottom=bottom_distance, ec='black', color=['yellow', 'orange'], label=['z1', 'z2'])
        # ax.set(xlabel="Activations of hidden layers", title=f"Iteration {i:d}")
        # plt.legend()
        # plt.show()
    # # Zum Debuggen der Initialisierung:
    # # Alle Prädiktionen sind ungefähr null, daher sollte die Anfang-loss bei reg=0 etwa 0.5*norm2(y)^2/N betragen.
    # # Erhöht man anschließend reg, sollte auch die loss steigen.
    # # Dann mit 20 data points und reg=0 trainieren und null-loss erreichen.
    # # Danach Aktivierungshistogramme mit tanh plotten.
    # if i == 0:
    #   print('Initialisierungscheck: reg sollte zuerst auf 0 gesetzt werden')
    #   print('Anfang-loss: {0:.4f}, sollte etwa {1:.4f} sein'.format(f(W), 0.5*np.sum(y*y)/N))
    #   print('Jetzt reg erhöhen und erneut ausführen, die loss sollte ebenfalls steigen')
    #   wait = input("Press Enter to continue.")

# ── finale training loss (ohne Regularisierung) ────────────────────────────────────────────
residuals  = f(W, predict=True) - y
dataloss = 0.5 * np.dot(residuals, residuals) / N 
print(f"\nFinal training data loss: {dataloss:.6e}")

# ── falls P=2, Visualisierung der Realität und der Modellvorhersagen ─────────────────────────────────
fig3d, axes3d = plt.subplots(subplot_kw=dict(projection="3d"))
max_x *= 1.1
gran = max_x / 100
valsx = np.arange(-max_x, max_x, gran)
valsy = valsx
x_grid, y_grid = np.meshgrid(valsx, valsy)
valsall = np.array([x_grid.flatten(), y_grid.flatten()])

reality = nature(valsall).reshape(x_grid.shape)
axes3d.plot_surface(x_grid, y_grid, reality, label="Realität", color="green", alpha=0.2)
axes3d.scatter(X[:, 0], X[:, 1], y, s=4, edgecolors="black", label="Beobachtungen")

yhat = f(W, Xp=valsall.T, predict=True).reshape(x_grid.shape)
axes3d.plot_surface(x_grid, y_grid, yhat, label="gelerntes Modell", color="red", alpha=0.4)
axes3d.set(xlabel="x", ylabel="y", title=f"Two-hidden-layer ReLU NN  (data loss {dataloss:.6e})")
axes3d.legend()

# ── Trainingsstatistiken plotten ───────────────────────────────────────────────────────────
iters = np.arange(MAX_ITER)
fig, ax = plt.subplots()
ax.scatter(iters, losses, s=2)
ax.set(xlabel="iteration", ylabel="loss", title="training loss")
fig, ax = plt.subplots()
ax.scatter(iters, ratios, s=2)
ax.axhline(1e-3, color="red", linestyle="--", label="target ≈ 1e-3")
ax.set(xlabel="iteration",
    ylabel="if lower/higher than 1e-3 learning rate might be too low/high",
    title="Vratio of the parameter update magnitudes to the parameter value magnitudes")
plt.show()
from builtins import range
import numpy as np


def affine_forward(x, w, b):
  """Berechnet den Forward-Pass für eine affine Schicht.

  Eingaben:
  - x: Ein Array mit Dimensionen (N, D), entspricht einem Minibatch aus N Inputs
  - w: Ein Array mit Gewichten, Dimensionen (D, M)
  - b: Ein Array mit Bias-Werten, Dimensionen (M,)

  Gibt ein Tupel zurück bestehend aus:
  - out: Ausgabe, Dimensionen (N, M)
  - cache: (x, w, b)
  """
  out = None

  out = x @ w + b

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """Berechnet den Backward-Pass für eine affine Schicht.

  Eingaben:
  - dout: Upstream-Ableitung, Dimensionen (N, M)
  - cache: Tupel bestehend aus:
    - x: Inputs, Dimensionen (N, D)
    - w: Gewichte, Dimensionen (D, M)
    - b: Bias-Werte, Dimensionen (M,)

  Gibt ein Tupel zurück bestehend aus:
  - dx: Gradient bezüglich x, Dimensionen (N, D)
  - dw: Gradient bezüglich w, Dimensionen (D, M)
  - db: Gradient bezüglich b, Dimensionen (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None

  dx = dout @ w.T
  dw = x.T @ dout
  db = dout.sum(axis=0)

  return dx, dw, db


def relu_forward(x):
  """Berechnet den Forward-Pass für eine ReLU Schicht.

  Eingabe:
  - x: Eingaben beliebiger Dimensionen

  Gibt ein Tupel zurück bestehend aus:
  - out: Ausgabe, gleiche Dimensionen wie x
  - cache: x
  """
  out = None

  out = np.maximum(0, x)

  cache = x  # caching the mask instead does not bring speed advantage
  return out, cache


def relu_backward(dout, cache):
  """Berechnet den Forward-Pass für eine ReLU Schicht.

  Eingabe:
  - dout: Upstream-Ableitungen beliebiger Dimensionen
  - cache: Eingabe x, gleiche Dimensionen wie dout

  Gibt zurück:
  - dx: Gradient bezüglich x
  """
  dx, x = None, cache

  dx = dout * (x > 0)

  return dx


def l2_loss(x, y):
  """Berechnet loss und Gradient für die L2-loss.

  Eingaben:
  - x: Inputs, Dimensionen (N,1), wobei x[i] der Score (=die Prädiktion) für das i-te Input ist.
  - y: Vektor von Zielwerten, Dimensionen (N,), wobei y[i] der Zielwert für x[i] ist.

  Gibt ein Tupel zurück bestehend aus:
  - loss: Skalarer Wert
  - dx: Gradient der loss bezüglich x
  """
  loss, dx = None, None

  N = len(y)  # Anzahl der Samples

  residual = x.flatten() - y         # N,
  loss = 0.5 * np.dot(residual, residual) / N

  dx = residual.reshape(-1, 1) / N

  return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """Forward-Pass für Batch-Normalisierung.

  Während des Trainings werden Mittelwert und (unkorrigierte) Varianz aus den
  Minibatch-Statistiken berechnet und zur Normalisierung der Eingabedaten verwendet.
  Zusätzlich wird ein exponentiell gleitender Durchschnitt für Mittelwert und Varianz
  jeder Feature-Dimension geführt, der zur Normalisierung während der Testphase dient.

  In jedem Schritt werden diese gleitenden Mittelwerte wie folgt aktualisiert:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Eingabe:
  - x: Inputs mit Dimensionen (N, D)
  - gamma: Skalierungsparameter, Dimensionen (D,)
  - beta: Verschiebungsparameter, Dimensionen (D,)
  - bn_param: Dictionary mit folgenden keys:
    - mode: 'train' oder 'test'; erforderlich
    - eps: Konstante für numerische Stabilität
    - momentum: Konstante für gleitende Mittelwerte
    - running_mean: Laufender Mittelwert, Dimensionen (D,)
    - running_var: Laufende Varianz, Dimensionen (D,)

  Gibt ein Tupel zurück bestehend aus:
  - out: Dimensionen (N, D)
  - cache: Werte für den Backward-Pass
  """
  mode = bn_param["mode"]
  eps = bn_param.get("eps", 1e-5)
  momentum = bn_param.get("momentum", 0.9)

  N, D = x.shape
  running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == "train":
    mu = x.mean(axis=0)        # Mittelwert pro Feature D,
    # print('mu: D,', mu)
    var = x.var(axis=0)        # Batch Varianz pro Feature D,
    # print('var: D,', var)
    std = np.sqrt(var + eps)   # Batch Standardabweichung pro Feature
    x_hat = (x - mu) / std     # standardisierte Eingabe N,D
    # print('x_hat: N,D', x_hat)
    out = gamma * x_hat + beta  # skalieren und verschieben (ergibt "x_hat") N,D
    # print('out: N,D', out)

    shape = bn_param.get('shape', (N, D))              # Dimensionen für backprop
    axis = bn_param.get('axis', 0)                     # Achse für Summenbildung in backprop
    cache = x, mu, var, std, gamma, x_hat, shape, axis  # für backprop speichern

    if axis == 0:                                                    # if not batchnorm
      running_mean = momentum * running_mean + (1 - momentum) * mu  # Mittelwert aktualisieren
      running_var = momentum * running_var + (1 - momentum) * var  # Varianz aktualisieren

  elif mode == "test":
    x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * x_hat + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Speichere die aktualisierten laufenden Mittelwerte zurück in bn_param
  bn_param["running_mean"] = running_mean
  bn_param["running_var"] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """Backward-Pass für Batch-Normalisierung.

  Eingaben:
  - dout: Upstream-Ableitungen, Dimensionen (N, D)
  - cache: Zwischenspeicher aus batchnorm_forward

  Gibt ein Tupel zurück bestehend aus:
  - dx: Gradient bezüglich der Eingaben x, Dimensionen (N, D)
  - dgamma: Gradient bezüglich Skalierungsparameter gamma, Dimensionen (D,)
  - dbeta: Gradient bezüglich Verschiebungsparameter beta, Dimensionen (D,)
  """
  dx, dgamma, dbeta = None, None, None
  x, mu, var, std, gamma, x_hat, shape, axis = cache          # Cache entpacken

  # Schritt für Schritt, forward -> backward, zur Visualisierung:
  # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
  # mu = np.sum(x, axis=0) / N # D,
  # xmu = x-mu # N,D
  # xmusq = xmu**2 # N,D
  # var = np.sum(xmusq, axis=0) / N # D,
  # std = np.sqrt(var + eps)   # D,
  # x_hat = xmu / std # N,D
  # out = gamma * x_hat + beta # N,D

  # N = len(dout)
  # xmu = x - mu
  # dbeta = np.sum(dout, axis=0) # D,
  # dgamma = np.sum(dout * x_hat, axis=0)      # D,
  # dx_hat = dout * gamma    # N,D
  # dstd = -np.sum(dx_hat * xmu, axis=0)  / (std**2) # D,
  # dxmu = dx_hat / std # N,D
  # dvar = 0.5 * dstd / std # D,
  # dxmusq = dvar / N # N,D
  # dxmu += 2 * xmu * dxmusq # N,D
  # dx = dxmu # N,D
  # dmu = -np.sum(dxmu, axis=0) # D,
  # dx += dmu / N

  dbeta = dout.reshape(shape, order='F').sum(axis)
  dgamma = (dout * x_hat).reshape(shape, order='F').sum(axis)
  dx_hat = dout * gamma
  dstd = -np.sum(dx_hat * (x - mu), axis=0) / (std**2)
  dvar = 0.5 * dstd / std
  dx1 = dx_hat / std + 2 * (x - mu) * dvar / len(dout)
  dmu = -np.sum(dx1, axis=0)
  dx2 = dmu / len(dout)
  dx = dx1 + dx2

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """Laufzeit optimierter Backward-Pass für Batch-Normalisierung.
  Eingaben / Ausgaben: identisch zu batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None

  _, _, _, std, gamma, x_hat, shape, axis = cache
  def S(x): return x.sum(axis=0)                     # Hilfsfunktion zur Summenbildung

  dbeta = dout.reshape(shape, order='F').sum(axis)
  dgamma = (dout * x_hat).reshape(shape, order='F').sum(axis)

  dx = dout * gamma / (len(dout) * std)
  dx = len(dout) * dx - S(dx * x_hat) * x_hat - S(dx)

  return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
  """Forward-Pass für Layer-Normalisierung.

  Während Training und Test wird die Eingabe pro Input normalisiert (nicht pro Batch) und
  anschließend mit gamma und beta skaliert bzw. verschoben, dies ähnlich wie bei Batchnorm.

  Im Gegensatz zur Batch-Normalisierung ist das Verhalten in Training und Test identisch,
  da keine laufenden Mittelwerte gespeichert werden müssen.

  Eingabe:
  - x: Inputs mit Dimensionen (N, D)
  - gamma: Skalierungsparameter, Dimensionen (D,)
  - beta: Verschiebungsparameter, Dimensionen (D,)
  - ln_param: Dictionary mit:
      - eps: Konstante zur numerischen Stabilität

  Gibt ein Tupel zurück bestehend aus:
  - out: Dimensionen (N, D)
  - cache: Werte für Backward-Pass
  """
  out, cache = None, None
  eps = ln_param.get("eps", 1e-5)

  # identisch zu Batchnorm im Trainingsmodus, aber über andere Achse summieren
  bn_param = {"mode": "train", "axis": 1, **ln_param}
  [gamma, beta] = np.atleast_2d(gamma, beta)          # 2D sichern für Transponierung

  out, cache = batchnorm_forward(x.T, gamma.T, beta.T, bn_param)  # Batchnorm-Äquivalent
  out = out.T                                                    # zurücktransponieren

  return out, cache


def layernorm_backward(dout, cache):
  """Backward-Pass für Layer-Normalisierung.

  Eingaben:
  - dout: Upstream-Ableitungen, Dimensionen (N, D)
  - cache: Zwischenspeicher aus layernorm_forward

  Gibt ein Tupel zurück bestehend aus:
  - dx: Gradient bezüglich Input x, Dimensionen (N, D)
  - dgamma: Gradient bezüglich gamma, Dimensionen (D,)
  - dbeta: Gradient bezüglich beta, Dimensionen (D,)
  """
  dx, dgamma, dbeta = None, None, None

  dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)  # identisch zu Batchnorm-Backprop
  dx = dx.T                                                 # dx zurücktransponieren

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """Forward-Pass für inverted Dropout.
  p ist die Wahrscheinlichkeit, dass ein Neuron aktiviert bleibt.

  Eingaben:
  - x: Eingabedaten beliebiger Form
  - dropout_param: Dictionary mit:
    - p: Wahrscheinlichkeit, dass ein Neuron behalten wird
    - mode: 'train' oder 'test'
    - seed: Zufalls-Seed für Reproduzierbarkeit

  Ausgaben:
  - out: Array gleicher Form wie x
  - cache: (dropout_param, mask)
  """
  p, mode = dropout_param["p"], dropout_param["mode"]
  if "seed" in dropout_param:
    np.random.seed(dropout_param["seed"])

  mask = None
  out = None

  if mode == "train":
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
  elif mode == "test":
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """Backward-Pass für inverted Dropout.

  Eingaben:
  - dout: Upstream-Ableitungen beliebiger Form
  - cache: (dropout_param, mask) aus dropout_forward
  """
  dropout_param, mask = cache
  mode = dropout_param["mode"]

  dx = None

  if mode == "train":
    dx = dout * mask
  elif mode == "test":
    dx = dout

  return dx

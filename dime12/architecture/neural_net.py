from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class NeuralNetwork(object):
  """Klasse für ein multi-layer neuronales Netzwerk.

  Das Netzwerk enthält eine beliebige Anzahl von hidden Layers und ReLU-Nichtlinearitäten
  sowie eine L2-Verlustfunktion. 
  Zusätzlich werden Dropout sowie Batch-/Layer-Normalisierung optional implementiert. 
  Für ein Netzwerk mit L Schichten ergibt sich folgende Architektur:

  {affin - [Batch-/Layer-Norm] - ReLU - [Dropout]} x (L - 1) - affin - L2

  wobei Batch-/Layer-Normalisierung und Dropout optional sind 
  und der {...}-Block L - 1 Mal wiederholt wird.

  Lernbare Parameter werden im Dictionary self.params gespeichert und mithilfe von der Solver Klasse gelernt.
  """

  def __init__(
      self,
      hidden_dims,
      input_dim=2,
      dropout_keep_ratio=1,
      normalization=None,
      reg=0.0,
      dtype=np.float32,
      seed=None,
  ):
    """Initialisierung des Netzwerks.

    Eingaben:
    - hidden_dims: Eine Liste von int, die die Dimension jeder hidden layer angibt.
    - input_dim: Eine int, die die Dimension des Inputs (hier D, manchmal auch P genannt) angibt.
    - dropout_keep_ratio: Skalar zwischen 0 und 1, der die Stärke des Dropouts angibt.
    dropout_keep_ratio=1 bedeutet, dass das Netzwerk kein Dropout verwendet.
    - normalization: Welche Art von Normalisierung das Netzwerk verwenden soll. Gültige Werte
    sind "batchnorm", "layernorm" oder None für keine Normalisierung (Standard).
    - reg: Skalar, der die Stärke der L2 ("Tikhonov") -Regularisierung angibt.
    - dtype: Ein NumPy-Datentypobjekt; alle Berechnungen werden mit diesem Datentyp durchgeführt.
    float32 ist schneller, aber weniger genau als float64.
    - seed: Falls nicht None, wird dieser Zufalls-Seed an die Dropout-Schichten übergeben.
    """
    self.normalization = normalization
    self.use_dropout = dropout_keep_ratio != 1
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.rng = np.random.default_rng(1303)
    self.params = {}

    for l, (i, j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, 1])):
      self.params[f'W{l+1}'] = self.rng.standard_normal((i, j)) * np.sqrt(2.0 / (i + j))
      self.params[f'b{l+1}'] = np.zeros(j)

      if self.normalization and l < self.num_layers - 1:
        self.params[f'gamma{l+1}'] = np.ones(j)
        self.params[f'beta{l+1}'] = np.zeros(j)
    # del self.params[f'gamma{l+1}'], self.params[f'beta{l+1}'] # no batchnorm after last FC

    # Bei Verwendung von Dropout müssen wir ein dropout_param-Dictionary an jede
    # Dropout-Schicht übergeben, damit die Schicht die Dropout-Wahrscheinlichkeit
    # und den Modus (Training / Test) kennt. Dasselbe dropout_param kann an jede
    # Dropout-Schicht übergeben werden.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
      if seed is not None:
        self.dropout_param["seed"] = seed

    # Bei der Batch-Normalisierung werden laufende Mittelwerte und Varianzen verfolgt,
    # daher wird ein spezielles bn_param-Objekt an jede Batch-Normalisierungsschicht
    # übergeben. self.bn_params[0] wird an den Forward-Pass der ersten Batch-
    # Normalisierungsschicht übergeben, self.bn_params[1] an den Forward-Pass der zweiten
    # Batch-Normalisierungsschicht usw.
    self.bn_params = []
    if self.normalization == "batchnorm":
      self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
    if self.normalization == "layernorm":
      self.bn_params = [{} for i in range(self.num_layers - 1)]

    # Cast alle Parameter in den korrekten Datentyp um.
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """Berechnet die Loss und den Gradienten für das Netzwerk.

    Eingaben:
    - X: Array der Inputs mit den Dimensionen (N, D)
    - y: Array der Zielwerte mit der Dimension (N,). y[i] gibt den Zielwert für X[i] an.

    Rückgabewerte:
    Wenn y None ist, wird ein Forward-Pass im Testmodus ausgeführt und Folgendes zurückgegeben:
    - scores: Array mit der Dimension (N,), das die Prädiktionen (oft auch y_hat genannt) enthält, wobei
    scores[i] die Prädiktion für X[i] ist.

    Wenn y nicht None ist, wird ein Forward- und ein Backward-Pass im Trainingsmodus ausgeführt
    und ein Tupel zurückgegeben bestehend aus:
    - loss: Skalarwert, der die Loss angibt
    - grads: Dictionary mit denselben Schlüsseln wie self.params, das Parameternamen
    auf die Gradienten der Loss bezüglich dieser Parameter abbildet.
    """
    X = X.astype(self.dtype)
    mode = "test" if y is None else "train"

    # Setze den Trainings-/Test-Modus für die Batchnorm-Parameter und den Dropout-Parameter,
    # da sie sich während Training und Test unterschiedlich verhalten.
    if self.use_dropout:
      self.dropout_param["mode"] = mode
    if self.normalization == "batchnorm":
      for bn_param in self.bn_params:
        bn_param["mode"] = mode
    scores = None

    cache = {}

    for l in range(self.num_layers):
      keys = [f'W{l+1}', f'b{l+1}', f'gamma{l+1}', f'beta{l+1}']   # Liste der params
      w, b, gamma, beta = (self.params.get(k, None) for k in keys)  # get param vals

      bn = self.bn_params[l] if gamma is not None else None  # bn params if exist
      do = self.dropout_param if self.use_dropout else None  # do params if exist

      X, cache[l] = generic_forward(X, w, b, gamma, beta, bn, do, l == self.num_layers - 1)  # Generisches Forward-Pass

    scores = X

    # If test mode return early.
    if mode == "test":
      return scores

    loss, grads = 0.0, {}

    loss, dout = l2_loss(scores, y)
    loss += 0.5 * self.reg * np.sum([np.sum(W**2) for k, W in self.params.items() if 'W' in k])

    for l in reversed(range(self.num_layers)):
      dout, dW, db, dgamma, dbeta = generic_backward(dout, cache[l])

      grads[f'W{l+1}'] = dW + self.reg * self.params[f'W{l+1}']
      grads[f'b{l+1}'] = db

      if dgamma is not None and l < self.num_layers - 1:
        grads[f'gamma{l+1}'] = dgamma
        grads[f'beta{l+1}'] = dbeta

    return loss, grads

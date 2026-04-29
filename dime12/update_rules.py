import numpy as np

"""
Diese Datei implementiert verschiedene First-Order-Update-Regeln, die häufig
zum Training neuronaler Netze verwendet werden. Jede Update-Regel akzeptiert
die aktuellen Gewichte und den Gradienten der Verlustfunktion bezüglich dieser
Gewichte, und erzeugt den nächsten Satz von Gewichten. Jede Update-Regel hat
die gleiche Schnittstelle:

def update(w, dw, config=None):

Eingaben:
  - w: Ein Array mit den aktuellen Gewichten.
  - dw: Ein Array mit denselben Dimensionen wie w, das den Gradienten der
    Verlustfunktion bezüglich w enthält.
  - config: Ein Dictionary, das Hyperparameter wie Lernrate, Momentum usw.
    enthält. Wenn die Update-Regel Werte über mehrere Iterationen speichern
    muss, enthält config auch diese zwischengespeicherten Werte.

Gibt zurück:
  - next_w: Der nächste Gewichtensatz nach dem Update.
  - config: Das config-Dictionary für die nächste Iteration.

HINWEIS: Für die meisten Update-Regeln wird die Standard-Lernrate vermutlich
keine guten Ergebnisse liefern; die Standardwerte der anderen Hyperparameter
(z.B. für Adam und RMSProp) sollten jedoch für viele Probleme gut funktionieren.

Aus Effizienzgründen können Update-Regeln in-place arbeiten, wobei w direkt
verändert wird und next_w gleich w gesetzt wird.
"""


def sgd(w, dw, config=None):
  """
  Führt Gradient Descent durch.
  Wird zu SGD, wenn der Batch stochastisch ausgewählt wird.

  config-Format:
  - learning_rate: Skalar für die Lernrate.
  """
  if config is None:
    config = {}
  config.setdefault("learning_rate", 1e-2)

  w -= config["learning_rate"] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Führt Gradient Descent mit Momentum durch.

  config-Format:
  - learning_rate: Skalar für die Lernrate.
  - momentum: Skalar zwischen 0 und 1, der das Momentum angibt.
    (Eigentlich eher ein Reibungskoeffizient)
    momentum = 0 entspricht normalem SGD.
  - velocity: Ein Array in derselben Form wie w und dw, das einen
    gleitenden Durchschnitt der Gradienten speichert.
  """
  if config is None:
    config = {}
  config.setdefault("learning_rate", 1e-2)
  config.setdefault("momentum", 0.9)
  v = config.get("velocity", np.zeros_like(w))

  next_w = None

  # nag, Nesterov Augmented Momentum
  mu = config['momentum']
  v_prev = v
  v = mu * v - config['learning_rate'] * dw
  next_w = w - mu * v_prev + (1 + mu) * v

  # traditional momentum
  # v = config['momentum'] * v - config['learning_rate'] * dw
  # next_w = w + v

  config["velocity"] = v

  return next_w, config


def rmsprop(w, dw, config=None):
  """
  Verwendet die RMSProp-Update-Regel, die einen gleitenden Durchschnitt der
  quadrierten Gradienten verwendet, um adaptive Lernraten pro Parameter zu setzen.

  config-Format:
  - learning_rate: Skalar für die Lernrate.
  - decay_rate: Skalar zwischen 0 und 1 für die Abklingrate des
    Gradienten-Caches.
  - epsilon: Kleine Konstante zur numerischen Stabilität.
  - cache: Gleitender Durchschnitt der zweiten Momente der Gradienten.
  """
  if config is None:
    config = {}
  config.setdefault("learning_rate", 1e-2)
  config.setdefault("decay_rate", 0.99)
  config.setdefault("epsilon", 1e-8)
  config.setdefault("cache", np.zeros_like(w))

  next_w = None

  keys = ['learning_rate', 'decay_rate', 'epsilon', 'cache']  # Reihenfolge beachten
  lr, dr, eps, cache = (config.get(key) for key in keys)  # Reihenfolge beachten

  config['cache'] = dr * cache + (1 - dr) * dw**2         # Cache aktualisieren
  next_w = w - lr * dw / (np.sqrt(config['cache']) + eps)  # Gewichte aktualisieren

  return next_w, config


def adam(w, dw, config=None):
  """
  Verwendet die Adam-Update-Regel, die gleitende Mittelwerte sowohl des
  Gradienten als auch seines Quadrats sowie einen Bias-Korrekturterm enthält.

  config-Format:
  - learning_rate: Skalar für die Lernrate.
  - beta1: Abklingrate für den gleitenden Mittelwert des ersten Moments.
  - beta2: Abklingrate für den gleitenden Mittelwert des zweiten Moments.
  - epsilon: Kleine Konstante zur numerischen Stabilität.
  - m: Gleitender Mittelwert des Gradienten.
  - v: Gleitender Mittelwert des quadrierten Gradienten.
  - t: Iterationsnummer.
  """
  if config is None:
    config = {}
  config.setdefault("learning_rate", 1e-3)
  config.setdefault("beta1", 0.9)
  config.setdefault("beta2", 0.999)
  config.setdefault("epsilon", 1e-8)
  config.setdefault("m", np.zeros_like(w))
  config.setdefault("v", np.zeros_like(w))
  config.setdefault("t", 0)

  next_w = None

  keys = ['learning_rate', 'beta1', 'beta2', 'epsilon', 'm', 'v', 't']  # Reihenfolge beachten
  lr, beta1, beta2, eps, m, v, t = (config.get(k) for k in keys)  # Reihenfolge beachten

  config['t'] = t = t + 1                             # Iterationszähler
  config['m'] = m = beta1 * m + (1 - beta1) * dw      # Glättung des Gradienten (Momentum)
  mt = m / (1 - beta1**t)                             # Bias-Korrektur
  config['v'] = v = beta2 * v + (1 - beta2) * (dw**2)  # Glättung (wie bei RMSProp)
  vt = v / (1 - beta2**t)                             # Bias-Korrektur
  next_w = w - lr * mt / (np.sqrt(vt) + eps)          # Gewichts-Update

  return next_w, config

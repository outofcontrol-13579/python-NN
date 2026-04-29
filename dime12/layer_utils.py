from .layers import *


def affine_relu_forward(x, w, b):
  """Hilfsschicht, die eine affine Transformation gefolgt von einer ReLU durchführt.

  Eingaben:
  - x: Input der affinen Schicht
  - w, b: Gewichte der affinen Schicht

  Gibt ein Tupel zurück bestehend aus:
  - out: Ausgabe der ReLU
  - cache: Objekt für den Backward-Pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """Backward-Pass für die affine-ReLU-Hilfsschicht.
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def generic_forward(x, w, b, gamma=None, beta=None, bn_param=None, dropout_param=None, last=False):
  """Hilfsschicht, die eine affine Transformation, optional Batch-/Layer-Normalisierung,
  eine ReLU und optional Dropout durchführt.

  Eingaben:
  - x: Input der affinen Schicht
  - w, b: Gewichte der affinen Schicht
  - gamma, beta: Skalierungs- und Verschiebungsparameter für die Batch-Normalisierung. 
  Siehe https://arxiv.org/pdf/1502.03167 für die Definitionen.
  - bn_param: Dictionary mit benötigten Batch-Normalisierung-(BN) Parametern
  - dropout_param: Dictionary mit benötigten Dropout-Parametern
  - last: Gibt an, ob nur die affine forward ausgeführt werden soll (= keine ReLu, insb. für die letzte, "output" Schicht)

  Gibt ein Tupel zurück bestehend aus:
  - out: Ausgabe der ReLU oder des Dropouts
  - cache: Objekt für den Backward-Pass
  """
  # Initialize optional caches to None
  bn_cache, ln_cache, relu_cache, dropout_cache = None, None, None, None

  # Affine ist obligatorisch
  out, fc_cache = affine_forward(x, w, b)

  if not last:
    # Optionale Normalisierungsschicht
    # Wenn bn_param einen Modus (train | test) enthält, handelt es sich um Batch-Normalisierung,
    # andernfalls um Layer-Normalisierung
    if bn_param is not None:
      if 'mode' in bn_param:
        out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
      else:
        out, ln_cache = layernorm_forward(out, gamma, beta, bn_param)

    # ReLu ist obligatorisch
    out, relu_cache = relu_forward(out)

    # Optionaler Dropout
    if dropout_param is not None:
      out, dropout_cache = dropout_forward(out, dropout_param)

  # Bereite den Cache für den Backward-Pass vor
  cache = fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache

  return out, cache


def generic_backward(dout, cache):
  """Backward-Pass für die affine-(BN/LN?)-ReLU-(Dropout?)-Hilfsschicht.
  """
  # Initialisiere Normalisierungsparameter mit None
  dgamma, dbeta = None, None

  # Hole die vorbereiteten Caches aus dem Forward-Pass
  fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache = cache

  # Falls Dropout ausgeführt wurde
  if dropout_cache is not None:
    dout = dropout_backward(dout, dropout_cache)

  # Falls ReLu ausgeführt wurde
  if relu_cache is not None:
    dout = relu_backward(dout, relu_cache)

  # Falls eine Normalisierung ausgeführt wurde
  if bn_cache is not None:
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
  elif ln_cache is not None:
    dout, dgamma, dbeta = layernorm_backward(dout, ln_cache)

  # Affine Backward ist obligatorisch
  dx, dw, db = affine_backward(dout, fc_cache)

  return dx, dw, db, dgamma, dbeta

from __future__ import print_function, division

from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from dime12 import update_rules


class Solver(object):
  """
  Ein Solver kapselt die gesamte Logik, die für das Training von Regressionsmodellen
  notwendig ist. Der Solver führt Gradient Descent unter Verwendung
  verschiedener in update_rules.py definierter Update-Regeln aus.

  Der Solver akzeptiert sowohl Trainings- als auch Validierungsdaten,
  sodass er regelmäßig die Genauigkeit auf Trainings- und
  Validierungsdaten überprüfen kann, um Overfitting zu erkennen.

  Um ein Modell zu trainieren, wird zunächst eine Solver-Instanz erstellt, wobei
  Modell, Datensatz und verschiedene Optionen (Lernrate, Batch-Größe usw.) an den
  Konstruktor übergeben werden. Anschließend wird die Methode train() aufgerufen,
  um den Optimierungsprozess auszuführen und das Modell zu trainieren.

  Nachdem die train()-Methode beendet ist, enthält model.params die Parameter,
  die während des Trainings die beste Leistung auf dem Validierungsdatensatz erzielt haben.
  Zusätzlich enthält die Instanzvariable solver.loss_history eine Liste aller während
  des Trainings aufgetretenen Verlustwerte, und die Instanzvariablen
  solver.train_acc_history und solver.val_acc_history enthalten Listen der
  Genauigkeiten des Modells auf Trainings- und Validierungsdaten nach jeder Epoche.

  Ein Beispiel für die Verwendung sieht etwa wie folgt aus:

  data = {
    'X_train': # Trainingsinputs (Prädiktoren)
    'y_train': # Trainingsantworten
    'X_val': # Validierungsinputs (Prädiktoren)
    'y_val': # Validierungsantworten
  }
  model = MyModel(hidden_size=100, reg=10)
  solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-4,
                  },
                  lr_decay=0.95,
                  num_epochs=5, batch_size=200,
                  print_every=100)
  solver.train()


  Ein Solver arbeitet mit einem Modellobjekt, das der folgenden API entsprechen muss:

  - model.params muss ein Dictionary sein, das String-Parameternamen auf NumPy-Arrays
    abbildet, die die Parameterwerte enthalten.

  - model.loss(X, y) muss eine Funktion sein, die sowohl den Trainingsverlust und die
    Gradienten als auch die scores (Prädiktionen) im Testmodus berechnet, mit folgenden
    Eingaben und Ausgaben:

    Eingaben:
    - X: Array mit einem Minibatch von Inputs der Dimensionen (N, D)
    - y: Array von Antworten der Form (N,), wobei y[i] die Antwort für X[i] ist.

    Rückgabe:
    Falls y None ist, wird ein Forward-Pass im Testmodus ausgeführt und Folgendes zurückgegeben:
    - scores: Array der Form (N, ), das Prädiktionen für X enthält, wobei
      scores[i] die Prädiktion für X[i] angibt.

    Falls y nicht None ist, wird ein Trainings-Forward- und Backward-Pass ausgeführt
    und ein Tupel zurückgegeben:
    - loss: Skalarer Verlustwert
    - grads: Dictionary mit denselben Schlüsseln wie self.params, das die Gradienten
      der Loss bezüglich der jeweiligen Parameter enthält.
  """

  def __init__(self, model, data, **kwargs):
    """
    Erstelle eine neue Solver-Instanz.

    Erforderliche Argumente:
    - model: Ein Modellobjekt, das der oben beschriebenen API entspricht
    - data: Ein Dictionary mit Trainings- und Validierungsdaten, das enthält:
      'X_train': Array, Dimensionen (N_train, D), der Trainingsinputs (Prädiktoren)
      'X_val': Array, Dimensionen (N_val, D), der Validierungsinputs (Prädiktoren)
      'y_train': Array, Dimensionen (N_train,), der Antworten für die Trainingsinputs
      'y_val': Array, Dimensionen (N_val,), der Antworten für die Validierungsinputs

    Optionale Argumente:
    - update_rule: Ein String, der den Namen einer Update-Regel in update_rules.py angibt.
      Standard ist 'sgd'.
    - optim_config: Ein Dictionary mit Hyperparametern, die an die gewählte
      Update-Regel übergeben werden. Jede Update-Regel benötigt unterschiedliche
      Hyperparameter (siehe update_rules.py), aber alle benötigen einen
      'learning_rate'-Parameter, der immer vorhanden sein sollte.
    - lr_decay: Skalar für die Lernraten-Abnahme; nach jeder Epoche wird die
      Lernrate mit diesem Wert multipliziert.
    - batch_size: Größe der Minibatches zur Berechnung von Verlust und Gradient
      während des Trainings. None benutzt den gesamten Trainingsdatensatz (N_train) 
      bei jeder Iteration.
    - num_epochs: Anzahl der Epochen, die während des Trainings durchlaufen werden.
    - print_every: Ganzzahl; loss wird alle
      print_every Iterationen ausgegeben.
    - verbose: Boolescher Wert; wenn False, wird während des Trainings keine
      Ausgabe erzeugt.
    - num_train_samples: Anzahl der Trainingsbeispiele zur Überprüfung der
      Trainingsgenauigkeit; Standard ist 1000; None verwendet den gesamten
      Trainingsdatensatz.
    - num_val_samples: Anzahl der Validierungsbeispiele zur Überprüfung der
      Validierungsgenauigkeit; Standard ist None, wodurch der gesamte
      Validierungsdatensatz verwendet wird.
    - checkpoint_name: Falls nicht None, werden hier nach jeder Epoche
      Modell-Checkpoints gespeichert.
    """
    self.model = model
    self.X_train = data["X_train"]
    self.y_train = data["y_train"]
    self.X_val = data["X_val"]
    self.y_val = data["y_val"]

    # Keyword-Argumente entpacken
    self.update_rule = kwargs.pop("update_rule", "sgd")
    self.optim_config = kwargs.pop("optim_config", {})
    self.lr_decay = kwargs.pop("lr_decay", 1.0)
    self.batch_size = kwargs.pop("batch_size", 100)
    self.num_epochs = kwargs.pop("num_epochs", 10)
    self.num_train_samples = kwargs.pop("num_train_samples", 1000)
    self.num_val_samples = kwargs.pop("num_val_samples", None)

    self.checkpoint_name = kwargs.pop("checkpoint_name", None)
    self.print_every = kwargs.pop("print_every", 10)
    self.rng = model.rng
    self.verbose = kwargs.pop("verbose", True)

    # Fehler bei unbekannten Argumenten
    if len(kwargs) > 0:
      extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
      raise ValueError("Unrecognized arguments %s" % extra)

    # Sicherstellen, dass die Update-Regel existiert, und dann den String
    # -Namen durch die tatsächliche Funktion ersetzen
    if not hasattr(update_rules, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(update_rules, self.update_rule)

    self._reset()

  def _reset(self):
    """
    Richtet einige Verwaltungsvariablen für die Optimierung ein. Diese Methode
    sollte nicht manuell aufgerufen werden.
    """
    self.epoch = 0
    self.best_val_acc = 1e6
    self.best_params = {}
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    # Eine tiefe Kopie der optim_config für jeden Parameter erstellen
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.items()}
      self.optim_configs[p] = d

  def _step(self):
    """
    Ein einzelnes Gradienten-Update wird durchgeführt. Diese Methode wird von train()
    aufgerufen und sollte nicht manuell verwendet werden.
    """
    if self.batch_size is None:
      # Trainiere mit allen bereitgestellten Daten und ohne Stochastik
      X_batch = self.X_train
      y_batch = self.y_train
    else:
      # Erstelle ein Mini-Batch der Trainingsdaten
      num_train = self.X_train.shape[0]
      batch_mask = self.rng.choice(num_train, self.batch_size)
      X_batch = self.X_train[batch_mask]
      y_batch = self.y_train[batch_mask]

    # Berechne loss und Gradienten
    loss, grads = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss)

    # Führe ein Parameter-Update durch
    for p, w in self.model.params.items():
      dw = grads[p]
      config = self.optim_configs[p]
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config

  def _save_checkpoint(self):
    if self.checkpoint_name is None:
      return
    checkpoint = {
        "model": self.model,
        "update_rule": self.update_rule,
        "lr_decay": self.lr_decay,
        "optim_config": self.optim_config,
        "batch_size": self.batch_size,
        "num_train_samples": self.num_train_samples,
        "num_val_samples": self.num_val_samples,
        "epoch": self.epoch,
        "loss_history": self.loss_history,
        "train_acc_history": self.train_acc_history,
        "val_acc_history": self.val_acc_history,
    }
    filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
    if self.verbose:
      print('Saving checkpoint to "%s"' % filename)
    with open(filename, "wb") as f:
      pickle.dump(checkpoint, f)

  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    """
    Überprüft die Genauigkeit des Modells auf den bereitgestellten Daten.

    Eingaben:
    - X: Inputarray der Form (N, D)
    - y: Antwortarray der Form (N,)
    - num_samples: Falls nicht None, werden die Daten zufällig subsampled und das Modell
      nur auf num_samples Datenpunkten getestet.
    - batch_size: Teilt X und y in Batches dieser Größe auf, um
      den Speicherverbrauch zu reduzieren.

    Rückgabe:
    - acc: Skalar, der den L2-Verlust zwischen Prädiktion und Zielwert (Antwort) angibt
    """

    # Eventuell die Daten subsamplen
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      mask = self.rng.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    # Prädiktionen in Batches berechnen
    num_batches = N // batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in range(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end])
      y_pred.append(scores)
    y_pred = np.vstack(y_pred)
    residual = y_pred.flatten() - y         # N,
    acc = 0.5 * np.dot(residual, residual) / len(y)
    return acc

  def train(self):
    """
    Führt die Optimierung aus, um das Modell zu trainieren.
    """
    if self.batch_size is None:
      num_iterations = self.num_epochs
      iterations_per_epoch = 1
    else:
      num_train = self.X_train.shape[0]
      iterations_per_epoch = max(num_train // self.batch_size, 1)
      num_iterations = self.num_epochs * iterations_per_epoch

    for t in range(num_iterations):
      self._step()

      # Eventuell training loss ausgeben
      if self.verbose and t % self.print_every == 0:
        print(
            "(Iteration %d / %d) loss: %e"
            % (t + 1, num_iterations, self.loss_history[-1])
        )

      # Am Ende jeder Epoche den Epoch-Zähler erhöhen und die Lernrate absenken
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]["learning_rate"] *= self.lr_decay

      # Trainings- und Validierungsgenauigkeit in der ersten Iteration, der letzten
      # Iteration sowie am Ende jeder Epoche überprüfen
      first_it = t == 0
      last_it = t == num_iterations - 1
      if first_it or last_it or epoch_end:
        # train_acc = self.loss_history[-1] # hack: falls reg=0, instead of self.check_accuracy, to speed up
        train_acc = self.check_accuracy(
            self.X_train, self.y_train, num_samples=self.num_train_samples
        )
        # val_acc = train_acc # hack: instead of self.check_accuracy, to speed up
        val_acc = self.check_accuracy(
            self.X_val, self.y_val, num_samples=self.num_val_samples
        )
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        self._save_checkpoint()

        if self.verbose:
          print(
              "(Epoch %d / %d) train loss: %e; val loss: %e"
              % (self.epoch, self.num_epochs, train_acc, val_acc)
          )

        # Bestes Modell abspeichern
        if val_acc < self.best_val_acc:
          # if train_acc < self.best_val_acc: # to visualize overfitting
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.items():
            self.best_params[k] = v.copy()

    # Am Ende des Trainings die besten Parameter ins Modell übernehmen
    self.model.params = self.best_params

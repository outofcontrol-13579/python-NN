## Problem-Formulierung

Gesucht ist eine parametrisierte Abbildung

$$
f_\theta : \mathbb{R}^3 \rightarrow \mathbb{R}^2,
$$

welche $(I_d, I_q, W_{el})$ auf $(U_d, U_q)$ abbildet. Die Parameter $\theta$ werden für einen wählbaren Datensatz $\mathcal{D}$ durch Minimierung einer wählbaren Verlustfunktion $\mathcal{L}$ bestimmt:

$$
\theta^*
=
\underset{\theta}{\operatorname{arg\,min}}
\;
\frac{1}{N}
\sum_{i=1}^{N}
\ell
\left(
f_\theta
\left(
I_d^{(i)}, I_q^{(i)}, W_{el}^{(i)}
\right),
\left(
U_d^{(i)}, U_q^{(i)}
\right)
\right).
$$

Dabei bezeichnet $f_\theta$ die parametrisierte Regressionsfunktion mit den Parametern $\theta$, $N$ die Anzahl der Datenpunkte und $\ell$ die gewählte Verlustfunktion, welche die Abweichung zwischen der Modellvorhersage und den tatsächlichen Zielgrößen quantifiziert.

Für die parametrisierte Abbildung $f_\theta$ werden drei unterschiedliche Modellierungsansätze untersucht:

1. ein physikalisch motiviertes bilineares Modell auf Basis der stationären Spannungsgleichungen im (dq)-Koordinatensystem,

2. ein Multi-Layer Perceptron (MLP) als flexible Regressionsfunktion, und

3. ein physikalisch informiertes Residualmodell, das das bilineare physikalische Modell um ein MLP zur Modellierung verbleibender Abweichungen erweitert.

Die Parameter des bilinearen Modells werden mittels des konvexen QP-Solvers Clarabel, die neuronalen Modelle mittels PyTorch optimiert. Die Modelle werden anhand des RMSE auf einem unabhängigen Validierungsdatensatz verglichen. Als Baseline dient das physikalische Modell mit Datasheet-Parametern. Trainings- und Validierungsdaten stammen aus Prüfstandsmessungen.

## Ergebnisse und Diskussion

### 1. Baseline und Parameterschätzung

Das **physikalische Modell mit Datasheet-Parametern** erreicht auf dem Validierungsdatensatz folgende RMSE-Werte:  
** Train (datasheet) metrics (physical units) **  
 Ud: RMSE = 0.4992 | R^2 = 0.8993  
 Uq: RMSE = 0.5642 | R^2 = 0.8369  
** Val (datasheet) metrics (physical units) **  
 **Ud: RMSE = 0.3710 | R^2 = 0.9162**  
 **Uq: RMSE = 0.2933 | R^2 = 0.9610**  
Datasheet Rd=0.03000 ohm, Rq=0.03000 ohm, Ld=0.000050 H, Lq=0.000050 H, Psi=0.00420 Wb

Die **Parameterschätzung aus Messdaten mittels des bilinearen Modells** reduziert diese Werte deutlich:  
** Train (bil) metrics (physical units) **  
 Ud: RMSE = 0.1310 | R^2 = 0.9931  
 Uq: RMSE = 0.0958 | R^2 = 0.9953  
 ** Val (bil) metrics (physical units) **  
 **Ud: RMSE = 0.0956 | R^2 = 0.9944**  
 **Uq: RMSE = 0.0803 | R^2 = 0.9971**  
 Learned Rd=0.03395 ohm, Rq=0.03395 ohm, Ld=0.000070 H, Lq=0.000068 H, Psi=0.00379 Wb

### 2. MLP

Ein **MLP mit moderater Kapazität** (Anzahl an Modellparametern) erzielt:  
model = MLP(len(PREDICTOR_KEYS), [32, 32], len(RESPONSE_KEYS), layernorms=False, silu=False)  
 (Epoch 4 / 5) 4.97 seconds. train loss: 0.002640; val_loss: 0.001614  
 ** Train (mlp) metrics (physical units) **  
 Ud: RMSE = 0.0897 | R^2 = 0.9967  
 Uq: RMSE = 0.0628 | R^2 = 0.9980  
 ** Val (mlp) metrics (physical units) **  
 **Ud: RMSE = 0.0612 | R^2 = 0.9977**  
 **Uq: RMSE = 0.0579 | R^2 = 0.9985**

Ein deutlich größeres MLP verbessert die Ergebnisse nur marginal:
MLP[512, 512]lFalse_sFalse_lr4.3e-05_reg0.00357_bs64_ep15.pth  
 val loss: 0.0015585467872243546  
 ** Train (mlp) metrics (physical units) **  
 Ud: RMSE = 0.0880 | R^2 = 0.9969  
 Uq: RMSE = 0.0565 | R^2 = 0.9984  
 ** Val (mlp) metrics (physical units) **  
 Ud: RMSE = 0.0613 | R^2 = 0.9977  
 Uq: RMSE = 0.0559 | R^2 = 0.9986

Dies deutet darauf hin, dass die moderate Modellkapazität für die vorliegende Regressionsaufgabe bereits weitgehend ausreichend ist.

### 3. Physikalisch informiertes Residualmodell

Zuerst sollen zwei Fragen behandelt werden: Das Kannibalisierungsproblem zwischen PSM-Term und Residual-Term, und die Wahl einer geeigneten Kapazität des Residual-Netzwerks.

#### (i) Kannibalisierung zwischen PSM- und Residualterm:

Der PSM-Term ist eine lineare Funktion der Parameter `id, iq, om, om*id, om*iq`. Das `residual_net` ist ein MLP, das `id, iq, om` direkt als Eingaben erhält. Ein MLP selbst moderater Größe kann dieselbe bilineare Kombination nahezu exakt approximieren – die Multiplikation zweier Eingaben ist für ein MLP mit ausreichender Breite trivial. Daher hat die Gesamtvorhersage

```
u_pred = u_psm(R, Ld, Lq, Psi) + u_res(NN weights)
```

eine ganze Mannigfaltigkeit von Kombinationen aus (PSM-Parametern, NN-weights), die zu (nahezu) identischen u_pred-Werten und damit zu einem nahezu identischen Loss führen. Die Loss-Funktion allein definiert keine Präferenz für einen bestimmten Punkt auf dieser Mannigfaltigkeit: Die Wahl des resultierenden Parametersatzes hängt vielmehr von der Optimierungsdynamik ab, die unter anderem durch die Initialisierung, den Learning-Rate-Schedule und die Reihenfolge der Batches beeinflusst wird, und nicht davon, an welchem Punkt der PSM-Term seinen physikalisch korrekten Anteil an der Gesamtleistung erbringt.  
L2-Regularisierung (weight decay) auf nn_weight_params wirkt diesem Effekt teilweise entgegen (es bestraft die Gewichte der letzten linearen Schicht, die – da die Hidden Layers des MLPs LayerNorm verwenden und dadurch in etwa skalen-normalisiert sind – die Ausgangsamplitude ziemlich direkt steuern). Allerdings ist das ein indirektes Instrument, und es ist gleichzeitig die einzige allgemeine Absicherung gegen Overfitting. Eine Erhöhung der L2-Regularisierung zur Unterdrückung der „Kannibalisierung“ würde daher gleichzeitig die Fähigkeit des Residualmodells einschränken, reale Nichtlinearitäten, beispielsweise Sättigungseffekte oder Eisenverluste, abzubilden.

**Gewählte Lösung**: Der konvexe QP-Schätzer des ersten Ansatzes wird zur Verankerung des Residual-Netzwerks eingesetzt, um ein freies Abdriften der Modellparameter zu verhindern:

```python
R0, Ld0, Lq0, Psi0 = df_bil['coef'][['Rd','Ld','Lq','Psi']]  # from the QP fit

prior_loss = (
    ((model.psm.R   - R0)   / R0)  ** 2 +
    ((model.psm.Ld  - Ld0)  / Ld0) ** 2 +
    ((model.psm.Lq  - Lq0)  / Lq0) ** 2 +
    ((model.psm.Psi - Psi0) / Psi0) ** 2
)
loss = loss + lambda_prior * prior_loss
```

lambda_prior wird anhand der Validierungsleistung bestimmt.  
R, Ld, Lq, Psi sowie val_loss werden lambda_prior in einem sweep gegenübergestellt und es wird nach einem Kompromiss gesucht: Die Regularisierung soll stark genug sein, um die Konkurrenz zwischen PSM-Term und Residual-Netzwerk aufzulösen, aber schwach genug, damit das Residual-Netzwerk nicht daran gehindert wird, die tatsächlich vorhandenen Nichtlinearitäten (z. B. Sättigung usw.) abzubilden, für deren Modellierung es vorgesehen ist.

#### (ii) Bestimmung einer geeigneten Kapazität für das Residual-Netzwerk

Es werden kleine MLPs direkt auf dem Residual-Target trainiert und den Validierungs-Loss für verschiedene Netzwerkgrößen verglichen.
Dort, wo sich der Validierungs-Loss bei zunehmender Netzwerkgröße nicht mehr merklich verbessert, liegt ungefähr die Kapazität, die die tatsächliche Residualstruktur benötigt. Diese Einschätzung ist unabhängig von möglichen „Kannibalisierungseffekten“, da es in diesem Fall keinen Physikterm gibt, mit dem das Netzwerk konkurrieren kann.
Als Upper-bound für die benötigte Kapazität kann die Größe [32,32] des Standalone-MLPs aus dem zweiten Ansatz angesehen werden: Das Standalone-MLP musste sowohl die bilinearen Physikterme als auch die nichtlinearen Residuen in einem einzigen Netzwerk lernen. Das ist eine grundsätzlich schwierigere Aufgabe als die eigentliche Aufgabe des Residual-Netzwerks.

- angemessene Kapazität:  
  PHY_RES[24, 24]lTrue_sFalse_lr8.3e-05_reg0.00357_bs256_ep15.pth
  val loss: 0.0016882646644242891
  ** Train (res) metrics (physical units) **
  Ud: RMSE = 0.0946 | R^2 = 0.9964
  Uq: RMSE = 0.0640 | R^2 = 0.9979
  ** Val (res) metrics (physical units) **
  **Ud: RMSE = 0.0645 | R^2 = 0.9975**
  **Uq: RMSE = 0.0575 | R^2 = 0.9985**
  Learned R=0.03394 ohm, Ld=0.000071 H, Lq=0.000067 H, Psi=0.00379 Wb
  (Ähnliche Ergebnisse sind ebenfalls in dem lambda_prior sweep erzielt worden:
  === capacity = [24, 24] ===  
  === lambda_prior = 0.8 ===  
  (Epoch 13 / 15) 4.20 seconds. train loss: 0.002784; val_loss: 0.001682  
  R=0.03395 ohm, Ld=0.000071 H, Lq=0.000068 H, Psi=0.00379 Wb  
  --> New best val loss: 0.0017 — checkpoint saved.  
  PHY_RES[24, 24]lTrue_sFalse_lr4.3e-05_reg0.00357_bs256_ep15)

- Zum Vergleich - Sehr hohe Kapazität:  
  (Epoch 10 / 15) 26.75 seconds. train loss: 0.002484; val_loss: 0.001506  
  R=0.03395 ohm, Ld=0.000070 H, Lq=0.000067 H, Psi=0.00379 Wb  
  --> New best val loss: 0.0015 — checkpoint saved.  
  PHY_RES[1028, 1028, 1028]lTrue_sFalse_lr3.3e-05_reg0.00357_bs256_ep15.pth  
  ** Train (res) metrics (physical units) **  
  Ud: RMSE = 0.0926 | R^2 = 0.9965  
  Uq: RMSE = 0.0542 | R^2 = 0.9985  
  ** Val (res) metrics (physical units) **  
  Ud: RMSE = 0.0594 | R^2 = 0.9979  
  Uq: RMSE = 0.0556 | R^2 = 0.9986  
  Learned R=0.03395 ohm, Ld=0.000070 H, Lq=0.000067 H, Psi=0.00379 Wb

## Zusammenfassung

Für den arithmetischen Mittel der beiden RMSE-Werte auf den Validierungsdatensatz ergibt sich:

<div align="center">

**Datasheet-Baseline: 0.3322 V**
**Bilineares Modell: 0.0880 V**
**Residualmodell: 0.0610 V**

</div>

Die Parameterschätzung des bilinearen PSM-Modells aus Messdaten verbessert die Validierungsgenauigkeit gegenüber der Verwendung von Datasheet-Parametern deutlich (über 50 %).

Durch die Erweiterung des identifizierten physikalischen Modells um ein Residual-Netzwerk kann die Genauigkeit weiter leicht verbessert werden (über 25% gegenüber dem reinen PSM-Modell).

Dass das korrekt verankerte physikalisch informierte Modell selbst bei großem lambda_prior das reine PSM-Modell übertrifft, deutet darauf hin, dass das Residualmodell zusätzliche physikalisch relevante Effekte erfasst, die durch die bilinearen PSM-Gleichungen nicht abgebildet werden.

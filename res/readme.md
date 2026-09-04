## Problem-Formulierung

Gesucht ist eine parametrisierte Abbildung

$$
f_\theta : \mathbb{R}^3 \rightarrow \mathbb{R}^2,
$$

welche $(I_d, I_q, W_{el})$ auf $(U_d, U_q)$ abbildet. Die Parameter $\theta$ werden für einen wählbaren Datensatz $\mathcal{D}$ durch Minimierung einer wählbaren Verlustfunktion $\mathcal{L}$ bestimmt:

$$
\theta^*=
\arg\min_{\theta}
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{L}
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

Dabei bezeichnet $f_\theta$ die parametrisierte Regressionsfunktion mit den Parametern $\theta$, $N$ die Anzahl der Datenpunkte und $\mathcal{L}$ die gewählte Verlustfunktion, welche die Abweichung zwischen der Modellvorhersage und den tatsächlichen Zielgrößen quantifiziert.

**Für die parametrisierte Abbildung $f_\theta$ werden vier unterschiedliche Modellierungsansätze untersucht (code: f\_.py):**

1. ein physikalisch motiviertes bilineares Modell auf Basis der stationären Spannungsgleichungen im (dq)-Koordinatensystem,

2. ein Multi-Layer Perceptron (MLP) als flexible, jedoch kaum interpretierbare Regressionsfunktion,

3. ein physikalisch informiertes Residualmodell, das das bilineare physikalische Modell um ein MLP zur Modellierung verbleibender Abweichungen erweitert, und

4. ein Greybox-Modell mit neuronaler Modulation der physikalischen Parametern.

Die Modelle werden anhand des RMSE auf einem Testdatensatz verglichen, der weder für das Training noch für die Modellentwicklung verwendet wurde. Ein separater Validierungsdatensatz dient während der Entwicklung zur Überwachung des Trainingsverlaufs (z. B. Verlustkurven, Checkpoint-Auswahl). Trainings-, Validierungs- und Testdatensätze stammen aus Prüfstandsmessungen. Um Informationslecks durch die Autokorrelation benachbarter Messpunkte zu vermeiden, erfolgt die Aufteilung nicht Messpunkten-weise, sondern in zusammenhängenden Zeitblöcken (Chunks), die jeweils vollständig einem der drei Datensätze zugewiesen werden. Die neuronalen Modelle werden mittels PyTorch optimiert. Als Baseline dient das physikalische Modell mit Datasheet-Parametern.

## Ergebnisse und Diskussion

### 1. Baseline und Parameterschätzung

Das **physikalische Modell mit Datasheet-Parametern** erreicht folgende RMSE-Werte:  
** Train (datasheet) metrics (physical units) **
Ud: RMSE = 0.4731 | R^2 = 0.9047
Uq: RMSE = 0.5197 | R^2 = 0.8626
** Val (datasheet) metrics (physical units) **
Ud: RMSE = 0.4488 | R^2 = 0.8996
Uq: RMSE = 0.5020 | R^2 = 0.8710
** Test (datasheet) metrics (physical units) **
Ud: RMSE = 0.5096 | R^2 = 0.8963
Uq: RMSE = 0.5351 | R^2 = 0.8721
Datasheet R=0.03000 ohm, Ld=0.000050 H, Lq=0.000050 H, Psi=0.00420 Wb

Die **Parameterschätzung aus Messdaten mittels des bilinearen Modells** reduziert diese Werte deutlich:  
** Train (bil) metrics (physical units) **
Ud: RMSE = 0.1228 | R^2 = 0.9936
Uq: RMSE = 0.0909 | R^2 = 0.9958
** Val (bil) metrics (physical units) **
Ud: RMSE = 0.1227 | R^2 = 0.9925
Uq: RMSE = 0.0902 | R^2 = 0.9958
** Test (bil) metrics (physical units) **
Ud: RMSE = 0.1354 | R^2 = 0.9927
Uq: RMSE = 0.0999 | R^2 = 0.9955
Learned R=0.03362 ohm, Ld=0.000071 H, Lq=0.000068 H, Psi=0.00383 Wb

### 2. MLP

Ein **MLP mit moderater Kapazität** (Anzahl an Modellparametern) erzielt:  
MLP[32, 32]lFalse_sFalse_lr0.00043_reg0.00357_bs64_ep5.pth
test loss: 0.0031278587485064884
** Train (mlp) metrics (physical units) **
Ud: RMSE = 0.0838 | R^2 = 0.9970
Uq: RMSE = 0.0599 | R^2 = 0.9982
** Val (mlp) metrics (physical units) **
Ud: RMSE = 0.0822 | R^2 = 0.9966
Uq: RMSE = 0.0626 | R^2 = 0.9980
** Test (mlp) metrics (physical units) **
Ud: RMSE = 0.0910 | R^2 = 0.9967
Uq: RMSE = 0.0732 | R^2 = 0.9976

Ein deutlich größeres MLP verbessert die Ergebnisse nur marginal:  
MLP[512, 512]lFalse_sFalse_lr4.3e-05_reg0.00357_bs64_ep5.pth
test loss: 0.0029390
** Train (mlp) metrics (physical units) **
Ud: RMSE = 0.0830 | R^2 = 0.9971
Uq: RMSE = 0.0572 | R^2 = 0.9983
** Val (mlp) metrics (physical units) **
Ud: RMSE = 0.0809 | R^2 = 0.9967
Uq: RMSE = 0.0653 | R^2 = 0.9978
** Test (mlp) metrics (physical units) **
Ud: RMSE = 0.0907 | R^2 = 0.9967
Uq: RMSE = 0.0683 | R^2 = 0.9979

Dies deutet darauf hin, dass die moderate Modellkapazität für die vorliegende Regressionsaufgabe bereits weitgehend ausreichend ist.

### 3. Physikalisch informiertes Residualmodell

Zuerst sollen zwei Fragen behandelt werden: Das Kannibalisierungsproblem zwischen PSM-Term und Residual-Term, und die Wahl einer geeigneten Kapazität des Residual-Netzwerks.

#### (i) Kannibalisierung zwischen PSM- und Residualterm:

Der PSM-Term ist eine lineare Funktion der Parameter `id, iq, om, om*id, om*iq`. Der Residualterm ist ein MLP, das `id, iq, om` direkt als Eingaben erhält. Ein MLP selbst moderater Größe kann dieselbe bilineare Kombination nahezu exakt approximieren: die Multiplikation zweier Eingaben ist für ein MLP mit ausreichender Breite trivial. Daher hat die Gesamtvorhersage

```
u_pred = u_psm(R, Ld, Lq, Psi) + u_res(NN weights)
```

eine ganze Mannigfaltigkeit von Kombinationen aus (PSM-Parametern, NN-weights), die zu (nahezu) identischen u_pred-Werten und damit zu einem nahezu identischen Loss führen. Die Loss-Funktion allein definiert keine Präferenz für einen bestimmten Punkt auf dieser Mannigfaltigkeit: Die Wahl des resultierenden Parametersatzes hängt vielmehr von der Optimierungsdynamik ab, die unter anderem durch die Initialisierung, den Learning-Rate-Schedule und die Reihenfolge der Batches beeinflusst wird, und nicht davon, an welchem Punkt der PSM-Term seinen physikalisch korrekten Anteil an der Gesamtleistung erbringt.  
L2-Regularisierung (weight decay) auf nn_weight_params wirkt diesem Effekt teilweise entgegen (es bestraft die Gewichte der letzten linearen Schicht, die – da die Hidden Layers des MLPs LayerNorm verwenden und dadurch in etwa skalen-normalisiert sind – die Ausgangsamplitude ziemlich direkt steuern). Allerdings ist das ein indirektes Instrument, und es ist gleichzeitig die einzige allgemeine Absicherung gegen Overfitting. Eine Erhöhung der L2-Regularisierung zur Unterdrückung der „Kannibalisierung“ würde daher gleichzeitig die Fähigkeit des Residualmodells einschränken, reale Nichtlinearitäten, beispielsweise Sättigungseffekte oder Eisenverluste, abzubilden.

**Gewählte Lösung**: Der konvexe QP-Schätzer des ersten Ansatzes wird zur Verankerung des Residual-Netzwerks eingesetzt, um ein freies Abdriften der Modellparameter zu verhindern:

```python
prior_loss = (
    ((model.psm.R   - R0)   / R0)  ** 2 +
    ((model.psm.Ld  - Ld0)  / Ld0) ** 2 +
    ((model.psm.Lq  - Lq0)  / Lq0) ** 2 +
    ((model.psm.Psi - Psi0) / Psi0) ** 2
)
loss = loss + lambda_prior * prior_loss
```

lambda_prior wird anhand der Validierungsleistung bestimmt:  
R, Ld, Lq, Psi sowie val_loss werden lambda_prior in einem sweep gegenübergestellt und es wird nach einem Kompromiss gesucht: Der prior soll stark genug sein, um die Konkurrenz zwischen PSM-Term und Residual-Netzwerk aufzulösen, aber schwach genug, dass das Residual-Netzwerk nicht daran gehindert wird, die tatsächlich vorhandenen Nichtlinearitäten (z. B. Sättigung usw.) abzubilden, für deren Modellierung es vorgesehen ist.

#### (ii) Bestimmung einer geeigneten Kapazität für das Residual-Netzwerk

Es werden kleine MLPs direkt auf dem Residual-Target trainiert und den Validierungs-Loss für verschiedene Netzwerkgrößen verglichen.
Dort, wo sich der Validierungs-Loss bei zunehmender Netzwerkgröße nicht mehr merklich verbessert, liegt ungefähr die Kapazität, die die tatsächliche Residualstruktur benötigt. Diese Einschätzung ist unabhängig von möglichen „Kannibalisierungseffekten“, da es in diesem Fall keinen Physikterm gibt, mit dem das Netzwerk konkurrieren kann.
Als Upper-bound für die benötigte Kapazität kann die Größe [32,32] des Standalone-MLPs aus dem zweiten Ansatz angesehen werden: Das Standalone-MLP musste sowohl die bilinearen Physikterme als auch die nichtlinearen Residuen in einem einzigen Netzwerk lernen. Das ist eine grundsätzlich schwierigere Aufgabe als die eigentliche Aufgabe des Residual-Netzwerks.

- angemessene Kapazität:  
  PHY_RES[24, 24]lTrue_sFalse_lp0.8_lr8.3e-05_reg0.00357_bs256_ep15.pth
  test loss: 0.003224116383719125
  ** Train (res) metrics (physical units) **
  Ud: RMSE = 0.0840 | R^2 = 0.9970
  Uq: RMSE = 0.0608 | R^2 = 0.9981
  ** Val (res) metrics (physical units) **
  Ud: RMSE = 0.0826 | R^2 = 0.9966
  Uq: RMSE = 0.0599 | R^2 = 0.9982
  ** Test (res) metrics (physical units) **
  Ud: RMSE = 0.0920 | R^2 = 0.9966
  Uq: RMSE = 0.0748 | R^2 = 0.9975
  Learned R=0.03361 ohm, Ld=0.000071 H, Lq=0.000068 H, Psi=0.00382 Wb

- Zum Vergleich - Sehr hohe Kapazität:  
  PHY_RES[512, 512]lTrue_sFalse_lp0.8_lr3.3e-05_reg0.00357_bs256_ep15.pth
  test loss: 0.0029853
  ** Train (res) metrics (physical units) **
  Ud: RMSE = 0.0832 | R^2 = 0.9971
  Uq: RMSE = 0.0588 | R^2 = 0.9982
  ** Val (res) metrics (physical units) **
  Ud: RMSE = 0.0813 | R^2 = 0.9967
  Uq: RMSE = 0.0636 | R^2 = 0.9979
  ** Test (res) metrics (physical units) **
  Ud: RMSE = 0.0914 | R^2 = 0.9967
  Uq: RMSE = 0.0689 | R^2 = 0.9979
  Learned R=0.03361 ohm, Ld=0.000071 H, Lq=0.000068 H, Psi=0.00384 Wb

### 4. Greybox-Modell mit neuronaler Parametermodulation

Ein vierter Ansatz erweitert das bilineare PSM-Modell, indem R, Psi, Ld und Lq nicht mehr als Konstanten, sondern als kontextabhängige effektive Parameter modelliert werden:

$$
R_{eff}(z) = R_0\,(1 + r_{scale}\tanh(NN_R(z))), \qquad
\Psi_{eff}(z) = \Psi_0\,(1 + \psi_{scale}\tanh(NN_\Psi(z))),
$$

analog für $L_{d,eff}$ und $L_{q,eff}$.  
Ein gemeinsamer latenter Kontext $z$ aus $i_d, i_q$ und $\omega$ speist dabei separate Köpfe, sodass die Korrekturen eine gemeinsame Ursache (etwa Temperatur, die sowohl R als auch Psi beeinflusst) teilen, aber unterschiedlich reagieren können, z. B. Psi zusätzlich durch Kreuzsättigung mit $i_d$, Ld/Lq durch Sättigung mit $i_d/i_q$.  
Die Köpfe werden null-initialisiert, sodass das Modell beim Nominalparametersatz startet; die Skalenfaktoren begrenzen die maximale Modulation explizit.  
Die Basisparameter $R_0, L_{d0}, L_{q0}$ werden log-parametrisiert, sodass sie unter jedem Optimierungsschritt strikt positiv bleiben. Zudem sind sie selbst lernbar, sodass sie den systematischen, konstanten Anteil der Anpassung absorbieren können, während das Modulationsnetz frei bleibt, ausschließlich die tatsächlich kontextabhängige Abweichung abzubilden, für die es vorgesehen ist.
Die dominante physikalische Ursache der Drift, die Wicklungs-/Magnettemperatur, ist aufgrund mangelnder Messung nicht Teil des Kontexts und muss indirekt über $i_d, i_q, \omega$ erschlossen werden.

#### (i) Identifizierbarkeit:

$R_{eff}$ bleibt auch bei $\omega \approx 0$ identifizierbar, da es als einziger Term nicht durch $\omega$ skaliert wird und sich bereits aus $u_d = R_{eff} i_d$ im Stillstand bestimmen lässt. $L_{d,eff}, L_{q,eff}$ und $\Psi_{eff}$ hingegen werden für $\omega \to 0$ gemeinsam unsichtbar für den Loss: drei unbestimmte Korrekturen im selben Niedrigdrehzahlbereich. Zudem sind $L_{d,eff}$ und $\Psi_{eff}$ außerhalb von $\omega \approx 0$ nur durch ihre unterschiedliche $i_d$-Abhängigkeit, nicht algebraisch getrennt, da beide über denselben Term $\omega \cdot(\ldots)$ in $u_q$ eingehen und aus demselben Trunk stammen; das Netzwerk kann daher beliebige, gleich gut passende Aufteilungen zwischen beiden Köpfen finden.  
Begegnet wird dem durch (1) Ausschluss von Datenpunkten mit $\omega \approx 0$, (2) enge Begrenzung der Ld/Lq-Skalen deutlich unterhalb von $r_{scale}$, und (3) eine Post-hoc-Prüfung auf Antikorrelation zwischen $L_{d,eff}$ und $\Psi_{eff}$ entlang $1/i_d$ als Signatur einer tatsächlich auftretenden Entartung.  
Volle Modulationsflexibilität aller Parameter zerstört tendenziell die Identifizierbarkeit, da Fehler beliebig zwischen den Parametern verschoben werden können, ohne die Vorhersage zu ändern. Idealerweise sollten daher gezielt nur Parameter mit begründeter Unsicherheit moduliert, gut charakterisierte Größen hingegen bei Möglichkeit fix gehalten werden.

#### (ii) Verhältnis zum bilinearen Modell:

Da alle Köpfe null-initialisiert sind, ist der Punkt „keine Modulation" ($\tanh(0)=0$) exakt erreichbar und entspricht genau der bilinearen Lösung. Das bilineare Modell ist somit als Spezialfall im modulierten Modell enthalten, dessen erreichbarer Trainingsloss folglich garantiert nicht schlechter als das OLS-Optimum ist.

GREY24_rs1_ps1_lds0.15_lqs0.15_grey0.01_lr0.001_reg0.00357_bs256_ep15.pth
saved test loss: 0.0031127222596933797
** Train (grey) metrics (physical units) **
Ud: RMSE = 0.0861 | R^2 = 0.9968
Uq: RMSE = 0.0589 | R^2 = 0.9982
** Val (grey) metrics (physical units) **
Ud: RMSE = 0.0839 | R^2 = 0.9965
Uq: RMSE = 0.0601 | R^2 = 0.9981
** Test (grey) metrics (physical units) **
Ud: RMSE = 0.0933 | R^2 = 0.9965
Uq: RMSE = 0.0704 | R^2 = 0.9978

## Zusammenfassung

Eine 5-fache Kreuzvalidierung der Modelle (Code: f_cv.py) ergibt auf dem Testdatensatz folgende Ergebnisse:

Bilineares Modell:
Ud: RMSE = 0.1235 +/- 0.0245 | R^2 = 0.9931 +/- 0.0010
Uq: RMSE = 0.0926 +/- 0.0096 | R^2 = 0.9956 +/- 0.0011

MLP-Modell:
Ud: RMSE = 0.0854 +/- 0.0204 | R^2 = 0.9967 +/- 0.0007
Uq: RMSE = 0.0682 +/- 0.0080 | R^2 = 0.9975 +/- 0.0008

Residualmodell:
Ud: RMSE = 0.0884 +/- 0.0187 | R^2 = 0.9965 +/- 0.0006
Uq: RMSE = 0.0690 +/- 0.0065 | R^2 = 0.9975 +/- 0.0006

Greybox-modell:
Ud: RMSE = 0.0880 +/- 0.0197 | R^2 = 0.9965 +/- 0.0007
Uq: RMSE = 0.0716 +/- 0.0088 | R^2 = 0.9973 +/- 0.0010

Für das arithmetische Mittel der beiden RMSE-Werte ergibt sich:

<div align="center">

**Datasheet-Baseline: 0.5224 V**  
**Bilineares Modell: 0.1081 ± 0.0170 V**  
**MLP-Modell: 0.0768 ± 0.0133 V**  
**Residualmodell: 0.0787 ± 0.0121 V**
**Greybox-modell: 0.0798 ± 0.0132 V**

</div>

Die Parameterschätzung des bilinearen PSM-Modells aus Messdaten verbessert die Genauigkeit gegenüber der Verwendung von Datasheet-Parametern erheblich (über 50 %).

Durch die Erweiterung des physikalischen Modells um ein Residual-Netzwerk kann die Genauigkeit weiter leicht verbessert werden (über 25% gegenüber dem reinen PSM-Modell). Dass das korrekt verankerte physikalisch informierte Modell selbst bei großem lambda_prior das reine PSM-Modell übertrifft, deutet darauf hin, dass das Residualmodell zusätzliche physikalisch relevante Effekte erfasst, die durch die bilinearen PSM-Gleichungen nicht abgebildet werden.

Ebenso übertrifft die Genauigkeit des Greybox-Modells die des bilinearen Modells deutlich. Darüber hinaus zeigen $R_{eff}$ sowie $\Psi_{eff}$ physikalisch plausible Trends in Abhängigkeit von $\omega$, $i_d$ und $i_q$: ein Hinweis auf reale Parameterdrift, die durch die Kontextmerkmale erklärbar ist.

Sowohl das Residualmodell als auch das Greybox-Modell erreichen nahezu die Genauigkeit der reinen MLP-Modellierung, bieten dabei den Vorteil einer stärker physikalisch strukturierten und damit besser interpretierbaren Modellform. Darüber hinaus wird erwartet, dass die physikalische Strukturierung ein robusteres Extrapolationsverhalten außerhalb der Trainingsverteilung ermöglicht. Die physikalische Eindeutigkeit der gelernten Parameter ist aufgrund der diskutierten Identifizierbarkeitsprobleme jedoch nur eingeschränkt gegeben.

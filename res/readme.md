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

- Der erste Ansatz basiert auf einem **Multi-Layer Perceptron (MLP)**, welches eine flexible Approximation der Abbildung ermöglicht.

- Als zweiter Ansatz werden die **Spannungsgleichungen im stationären Zustand im (dq)-Koordinatensystem** betrachtet. Diese weisen näherungsweise eine bilineare Struktur auf und stellen eine physikalisch motivierte Modellierung der Abbildung dar.

- Der dritte Ansatz basiert auf einem **physikalisch informierten Residualmodell**. Hierbei wird das physikalisch motivierte PSM-Modell von dem zweiten Ansatz mit einem MLP, als Residualmodell, kombiniert.

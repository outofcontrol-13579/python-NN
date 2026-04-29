Um alle erforderlichen Pakete zu installieren:
```bash
pip install -r requirements.txt
```

- main_reg.py: fit eine 2D-Sinc-Funktion mit einem neuronalen Netzwerk.
- example.py: 2000 Prädiktoren, davon nur wenige informativ. Lineare Regression vs. neuronales Netzwerk mit Regularisierung und Dropout.
- example_reg2D: 2 Prädiktoren. Basis-Erweiterung (lineares Modell) vs. neuronales Netzwerk. Kann zur Visualisierung von Overfitting verwendet werden.

/dime12  
Flexible NN-Modell- und Solver-Klasse

/toy  
- reg.py: standalone 2-layers NN für Regression, nutzbar mit Autograd.
Liefert die gleichen Ergebnisse wie main_reg.py, wenn die gleichen Parameter verwendet werden. 

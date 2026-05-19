## Quickstart in VSCode:  
Strg-Shift-p, Python: Create Environment, Quick Create venv  
  
-> dies installiert die erforderlichen Pakete, die in requirements.txt bereits definiert sind. Die .py files sollten dann direkt laufen.  

## Kurze Inhaltsbeschreibung:  
- main_reg.py: fit eine 2D-Sinc-Funktion mit einem neuronalen Netzwerk.
- example.py: 2000 Prädiktoren, davon nur wenige informativ. Lineare Regression vs. neuronales Netzwerk mit Regularisierung und Dropout.
- example_reg2D: 2 Prädiktoren. Basis-Erweiterung (lineares Modell) vs. neuronales Netzwerk. Kann zur Visualisierung von Overfitting verwendet werden.

/dime12  
Flexible NN-Modell- und Solver-Klasse

/toy  
- reg.py: standalone 2-layers NN für Regression, nutzbar mit Autograd.
Liefert die gleichen Ergebnisse wie main_reg.py, wenn die gleichen Parameter verwendet werden. 

/param-ident
- motor_batches.py: solves a QP with equality and unequality constraints for batches of measurements
- motor_recursive.py: relaxes the unequality constraints and subsums the equality constraints into the objective. This allows a simpler OLS formulation in a reduced space and thus a rank-2 recursive update (Recursive Least Square) can be used for each measurement sample to speed up computation.

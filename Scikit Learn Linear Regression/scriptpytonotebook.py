import nbformat
from nbformat.v4 import new_notebook, new_code_cell

with open("regresion_1.py", "r") as f:
    code = f.read()

nb = new_notebook(cells=[new_code_cell(code)])

with open("tips_Ismael_Garcia.ipynb", "w") as f:
    nbformat.write(nb, f)

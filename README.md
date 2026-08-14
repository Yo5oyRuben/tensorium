# tensorium

`tensorium` is a small symbolic tensor-calculus library built on top of
[SymPy](https://www.sympy.org/). It was designed as an experimental and
educational tool for working with tensor fields, covariant derivatives and
connections in local coordinates.

The project was developed while studying different symbolic-computation
approaches to differential geometry. Its main goal is to provide a
compact Python implementation whose internal representation is easy to inspect,
modify and extend.

## Features

`tensorium` currently supports:

- smooth manifolds, open sets, charts and coordinate changes;
- local and global tensor fields with explicit index variance;
- scalar fields, vector fields and one-forms as tensor-field specializations;
- tensor products, contractions, tensor evaluation and basic tensor algebra;
- metric manifolds, index raising/lowering and Levi-Civita connections;
- covariant derivatives of arbitrary tensor fields;
- Riemann and Ricci curvature computations;
- tensor fields with finite-dimensional internal degrees of freedom;
- gauge connections and mixed affine/gauge covariant derivatives;
- symbolic tensor operators, operator composition, commutators and internal actions;
- notebook-friendly mathematical display utilities.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/your-user/tensorium.git
cd tensorium
pip install -e .
```

The only required symbolic backend is SymPy. For notebook use, install Jupyter
or use the Python/Jupyter support included in your preferred editor.

## Tutorials

The `notebooks/` folder contains a guided introduction:

- `tensorium_01_basics.ipynb`: manifolds, charts, scalar fields, vector fields,
  one-forms and tensor algebra;
- `tensorium_02_metric_geometry.ipynb`: metric manifolds, Levi-Civita
  connections, curvature and affine covariant derivatives;
- `tensorium_03_internal_fields.ipynb`: tensor fields with internal degrees of
  freedom and matrix-valued fields;
- `tensorium_04_gauge_covariant_derivatives.ipynb`: gauge connections and
  covariant derivatives combining geometric and internal structure;
- `tensorium_05_tensor_operators.ipynb`: tensor operators, composition,
  contractions, commutators and Dirac-like operators;

## License

This project is distributed under the MIT License.

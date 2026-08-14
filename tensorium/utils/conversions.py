import symengine as se
from sympy.core import sympify

def _to_symengine(expr):
    """Convert an expression to a SymEngine object via its string form."""
    return se.sympify(str(expr))

def _to_sympy(expr, local_dict=None):
    """Convert an expression to SymPy, optionally using a local symbol map."""
    if expr==0: return sympify(0)
    if local_dict is None: return sympify(expr)
    return sympify(str(expr), locals=local_dict)

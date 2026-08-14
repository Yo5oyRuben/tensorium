from sympy import sympify
import symengine as se

def _simplify_expr_symengine(expr, locals_dict=None):
    """Simplify a SymPy expression with SymEngine and return a SymPy expression."""
    if se is None: return expr
    se_expr=se.sympify(str(expr))
    se_expr=se_expr.simplify()
    num,den=se_expr.as_numer_denom()
    if den!=1: se_expr=(num.simplify()/den.simplify()).simplify()
    if locals_dict is None: return sympify(str(se_expr))
    return sympify(str(se_expr), locals=locals_dict)
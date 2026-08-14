from sympy import count_ops, simplify, sympify

try:
    import symengine as se
except ImportError:  # pragma: no cover - optional dependency
    se = None


def _to_sympy(expr):
    """Convert an expression to SymPy."""
    return sympify(expr)


def _to_symengine(expr):
    """Convert an expression to SymEngine."""
    if se is None: raise ImportError("symengine is not installed")
    return se.sympify(str(expr))


def simplify_symengine_expr(expr, replacements=(), rounds=2):
    """Simplify one SymEngine expression and return a SymEngine expression."""
    if se is None: raise ImportError("symengine is not installed")
    current=expr
    for old,new in replacements: current=current.subs(old,new)
    for _ in range(rounds):
        previous=current
        current=current.simplify()
        if current==previous: break
    for old,new in reversed(tuple(replacements)): current=current.subs(new,old)
    return current.simplify()


def maybe_simplify_symengine_expr(expr, replacements=(), min_complexity=20, rounds=1):
    """Simplify with SymEngine only above a rough complexity threshold."""
    try: complexity=count_ops(_to_sympy(expr))
    except Exception: complexity=None
    if complexity is not None and complexity<min_complexity: return expr
    return simplify_symengine_expr(expr,replacements=replacements,rounds=rounds)


def simplify_sage_exprs(exprs, sage_python=None):
    """Compatibility fallback for Sage simplification hooks."""
    return [simplify(_to_sympy(expr)) for expr in exprs]


def build_metric_replacements(metric_entries, inverse_entries, max_replacements=12, min_complexity=4):
    """Return metric replacements used by older benchmark routines."""
    return []

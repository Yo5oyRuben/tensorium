from sympy.matrices import ImmutableDenseMatrix
from tensorium.core.manifold import OpenSet
from sympy.core import Basic, Tuple, Symbol, Dict
from sympy import solve, Dummy, sympify
from sympy.core.symbol import Str
from tensorium.utils.conversions import _to_symengine, _to_sympy
import symengine as se
from sympy import diff

class CoordinateSymbol(Symbol):
    """Represent a coordinate symbol attached to a specific chart.
    Unlike a plain SymPy symbol, this object remembers both the chart it
    comes from and the coordinate index it represents.
    """
    def __new__(cls, chart, index, **assumptions):
        """Create a coordinate symbol from a chart coordinate definition.
        Parameters:
         - chart : Chart (Chart that owns the coordinate)
         - index : int (Position of the coordinate inside the chart)
         - **assumptions: Standard SymPy symbol assumptions.
        """
        name = chart.args[2][index].name
        obj = super().__new__(cls, name, **assumptions)
        obj._chart=chart
        obj._index=index
        return obj
    
    @property
    def chart(self):
        """Return the chart that owns this coordinate symbol."""
        return self._chart
    @property
    def index(self):
        """Return the coordinate position inside the chart."""
        return self._index

    def __getnewargs__(self): 
        """Return the constructor arguments required to rebuild the symbol."""
        return (self.chart, self.index)
    def _hashable_content(self):
        """Return hashable data that distinguishes this coordinate symbol."""
        return (self.chart, self.index) + tuple(sorted(self.assumptions0.items()))
    def transform_to(self, other_chart):
        """Express this coordinate in another chart.
        Parameters:
         - other_chart : Chart (Target chart in which the coordinate should be expressed)
        Returns:
         - Expr: Expression for this coordinate after transformation to ``other_chart``.
        """
        return self.chart.transform(other_chart)[self.index]


class Chart(Basic):
    """Represent a coordinate chart on an open set.
    A chart stores its coordinate symbols and any known transition maps to
    other charts. Transition relations are expressed as coordinates of the
    target chart written as functions of the coordinates of this chart.
    """
    def __new__(cls, name, open_set, symbols=None, relations=None):
        """Create a chart with coordinates and optional transition relations.
        Parameters:
         - name : str or Str (Symbolic name of the chart)
         - open_set : OpenSet (Open set on which the chart is defined)
         - symbols : iterable of Symbol, optional
            Coordinate symbols for the chart. If omitted, default real symbols
            are generated automatically.
        relations : mapping, optional
            Dictionary whose keys are other charts and whose values are tuples
            of coordinate expressions giving the target coordinates as functions
            of this chart's coordinates.
        """
        if not isinstance(name, Str): name = Str(name)
        if relations is None: relations = {}
        if symbols is None:
            symbols = Tuple(*[Symbol(f"{name.name}_{i}", real=True) for i in range(open_set.dim)])
        else: symbols=Tuple(*[Symbol(s.name, **s._assumptions.generator) for s in symbols])
        if len(symbols)!=open_set.dim:
            raise ValueError("The number of symbols specified doesn't match"
                            "the dimension of the open set")
        rel_temp = {}
        for other_chart, exprs in relations.items():
            if len(exprs)!=open_set.dim:
                raise ValueError(f"Expected {open_set.dim} coordinate expressions "
                                 "in the chart relation," f"got {len(exprs)}.")
            rel_temp[other_chart]=Tuple(*exprs)
        relations=Dict(rel_temp)
        obj = super().__new__(cls, name, open_set, symbols, relations)
        obj._symbols=tuple(CoordinateSymbol(obj,i,**s._assumptions.generator) 
                     for i,s in enumerate(symbols))
        obj._inverse_transform_cache={}
        obj._jacobian_cache={}
        obj._inverse_jacobian_cache={}
        obj._transform_cache={}
        return obj
    
    @property
    def name(self):
        """Return the symbolic chart name."""
        return self.args[0]
    @property
    def open_set(self):
        """Return the open set where the chart is defined."""
        return self.args[1]
    @property
    def dim(self):
        """Return the dimension of the chart."""
        return self.open_set.dim
    @property
    def manifold(self):
        """Return the manifold of the chart."""
        return self.open_set.manifold
    @property
    def relations(self):
        """Return the known transition relations from this chart."""
        return self.args[3]
    @property 
    def symbols(self):
        """Return the coordinate symbols associated with this chart.
        Each symbol is returned as a ``CoordinateSymbol`` so it keeps track of
        its originating chart and coordinate index.
        """
        return self._symbols
    
    def restrict(self, W):
        """Restrict the chart to another open set. Parameters:
         - W : OpenSet
            Open set used to restrict the domain of the chart. It must belong
            to the same manifold.
        Returns:
         - Chart
            A new chart defined on the intersection of the current chart domain
            and ``W``.
        """
        if not isinstance(W, OpenSet): raise ValueError("W must be an open set")
        if W.manifold!=self.manifold: raise ValueError("W must belong to the same manifold as the chart")
        new_name = Str(f"{self.name}|_{{{W.name}}}")
        return Chart(new_name, self.open_set.intersection(W), self.args[2], self.relations)
    
    def base_scalar(self, i):
        """Return the i-th coordinate function of the chart basis."""
        return self.symbols[i]

    def base_scalars(self):
        """Return all coordinate functions of the chart basis."""
        return list(self.symbols)

    def _solve_inverse(self, other_chart):
        """Compute the inverse transformation from another chart to this chart.
        This method is used when the direct relation from this chart to
        ``other_chart`` is not known, but the reverse relation is available.
        Parameters:
         - other_chart : Chart (Chart whose coordinates are related to this chart)
        Returns:
         - Tuple: Tuple of expressions giving this chart's coordinates in terms of
            ``other_chart`` coordinates.
        """
        if other_chart in self._inverse_transform_cache:
            return self._inverse_transform_cache[other_chart]
        dummies=[Dummy() for _ in range(self.dim)]
        subs_rule=list(zip(self.symbols, dummies))+list(zip(self.args[2], dummies))
        exprs_dummy=[expr.subs(subs_rule) for expr in self.relations[other_chart]]
        sols = solve([expr-sym for expr,sym in zip(exprs_dummy, other_chart.symbols)],
            dummies, dict=True)
        if len(sols)==0: raise NotImplementedError("Cannot solve inverse transformation.")
        if len(sols)>1: raise ValueError("Inverse transformation is not unique.")
        sol=sols[0]
        result=Tuple(*[sol[d] for d in dummies])
        self._inverse_transform_cache[other_chart]=result
        return result

    def _direct_transform(self, other_chart):
        """Return a direct transformation to another chart when it is immediately available.
        Parameters
        - other_chart : Chart (Target chart
        Returns:
        - Tuple: 
            Coordinate expressions for ``other_chart`` written in terms of this
            chart's coordinates.
        """
        if self==other_chart: return Tuple(*self.symbols)
        if other_chart in self.relations: return self.relations[other_chart]
        if self in other_chart.relations: return other_chart._solve_inverse(self)
        raise KeyError("Direct transformation not known")
    
    def transform(self, other_chart):
        """Return the coordinate transformation from this chart to another chart.
        Parameters:
         - other_chart : Chart (Target chart)
        Returns:
         - Tuple:
            Coordinate expressions for ``other_chart`` in terms of this chart's
            coordinates.
        Notes:
        If the transformation is not directly available, the method asks the
        manifold atlas to compose a path of known chart transitions."""
        if not isinstance(other_chart, Chart): raise ValueError("other_chart must be a Chart")
        if other_chart.manifold!=self.manifold: raise ValueError("The two charts are not defined on the same manifold")

        if other_chart in self._transform_cache: return self._transform_cache[other_chart]
        if self==other_chart:
            res=Tuple(*self.symbols)
            self._transform_cache[other_chart]=res
            return res
        
        try: res=self._direct_transform(other_chart)
        except KeyError:
            if self.manifold.has_atlas():
                return self.manifold.atlas.transform(self, other_chart)
            raise KeyError("Transformation not known")
        self._transform_cache[other_chart]=res
        return res
    
    def jacobian(self, other_chart):
        """Return the Jacobian matrix of the transformation to another chart.
        Parameters:
         - other_chart : Chart (Target chart)
        Returns:
         - ImmutableDenseMatrix
            Jacobian of the coordinate transformation with respect to this
            chart's coordinates.
        Notes:
        If the manifold has an atlas, the atlas-level cached Jacobian is used.
        Otherwise the Jacobian is computed directly from ``transform``.
        """
        if not isinstance(other_chart, Chart): raise ValueError("other_chart must be a Chart")
        if other_chart.manifold!=self.manifold: raise ValueError("The two charts are not defined on the same manifold")

        if other_chart in self._jacobian_cache: return self._jacobian_cache[other_chart]
        if self==other_chart:
            res=ImmutableDenseMatrix(self.dim, self.dim, [1 if i==j else 0 for i in range(self.dim) for j in range(self.dim)])
            self._jacobian_cache[other_chart]=res
            return res
        exprs=self.transform(other_chart)
        free_syms=set(self.symbols)|set(other_chart.symbols)
        for expr in exprs: free_syms.update(sympify(expr).free_symbols)
        locals_dict={str(s): s for s in free_syms}
        se_exprs=[_to_symengine(expr) for expr in exprs]
        se_symbols=[_to_symengine(sym) for sym in self.symbols]
        rows=[]
        for expr in se_exprs:
            row=[]
            for sym in se_symbols:
                deriv=(se.diff(expr,sym)).simplify()
                row.append(_to_sympy(deriv, local_dict=locals_dict))
            rows.append(row)
        res=ImmutableDenseMatrix(rows)
        self._jacobian_cache[other_chart]=res
        return res


def chart_diff(expr, chart, index):
    """Differentiate an expression with respect to a chart coordinate.

    The library currently supports two equivalent ways of writing component
    expressions:

    - with the plain SymPy symbols originally passed to ``Chart(..., symbols=)``
      and stored in ``chart.args[2]``;
    - with the wrapped ``CoordinateSymbol`` objects exposed by
      ``chart.symbols``.

    This helper makes the differential routines robust to both conventions.
    """
    plain = chart.args[2][index]
    wrapped = chart.symbols[index]

    free = getattr(expr, "free_symbols", set())
    has_plain = plain in free
    has_wrapped = wrapped in free

    if has_plain and has_wrapped:
        # Treat both symbol families as the same coordinate and avoid double
        # counting if a mixed expression ever appears.
        return diff(expr.subs({wrapped: plain}), plain)
    if has_plain:
        return diff(expr, plain)
    if has_wrapped:
        return diff(expr, wrapped)
    return 0

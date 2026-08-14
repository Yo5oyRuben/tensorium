from sympy.core import Basic, Tuple
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify

class Manifold(Basic):
    """Represent an abstract differentiable manifold.
    A manifold is identified by a symbolic name and an integer dimension.
    It can optionally store an atlas, which provides charts and transition
    maps used to move between coordinate systems.
    """
    def __new__(cls, name, dim):
        """Create a manifold. Parameters:
         - name : str or Str. (Name used to identify the manifold symbolically)
         - dim : int or sympifiable expression. (Dimension of the manifold)
        The atlas is initialized as ``None`` and can be attached later with ``set_atlas``.
        """
        if not isinstance(name, Str): name=Str(name)
        dim=_sympify(dim)
        obj=super().__new__(cls, name, dim)
        obj._atlas=None
        return obj
    @property
    def name(self):
        """Return the symbolic manifold name."""
        return self.args[0]
    @property
    def dim(self):
        """Return the manifold dimension."""
        return self.args[1]
    @property
    def atlas(self):
        """Return the atlas attached to the manifold, if any."""
        return self._atlas
    
    def set_atlas(self, atlas):
        """Attach an atlas to the manifold after verifying ownership.
         - atlas : Atlas (Atlas to associate with this manifold)"""
        if(self!=atlas.manifold):
            raise ValueError("This atlas does not belong to this manifold")
        self._atlas=atlas
    
    def has_atlas(self):
        """Return True when the manifold already has an atlas attached."""
        if self.atlas is None: return False
        else: return True

class OpenSet(Basic):
    """Represent an open set of a manifold.
    An open set is identified by a name and the manifold it belongs to.
    It may also keep track of factor open sets, which are used internally
    to describe intersections in a structured way.
    """
    def __new__(cls, name, manifold, factors=None):
        """Create an open set on a manifold. Parameters:
         - name : str or Str (Symbolic name of the open set)
         - manifold : Manifold (Manifold to which the open set belongs)
         - factors : iterable of OpenSet, optional (Collection of component open sets used to describe this set)
        """
        if not isinstance(name, Str): name=Str(name)
        if factors is None: return super().__new__(cls, name, manifold, None)
        else: return super().__new__(cls, name, manifold, Tuple(*factors))
    
    @property
    def name(self):
        """Return the symbolic name of the open set."""
        return self.args[0]
    @property
    def manifold(self):
        """Return the manifold to which the open set belongs."""
        return self.args[1]
    @property
    def factors(self):
        """Return the factor open sets used to describe this set."""
        return self.args[2]
    @property
    def dim(self):
        """Return the dimension inherited from the manifold."""
        return self.manifold.dim

    @property
    def _effective_factors(self):
        """Return the tuple of factors that effectively define this open set.
        If the set has no explicit factorization, it is treated as its own
        single effective factor.
        """
        if self.factors is None: return (self,)
        else: return tuple(self.factors)

    def intersection(self, other):
        """Return the intersection with another open set. Parameters:
         - other : OpenSet (Open set to intersect with this one. 
                        It must belong to the samemanifold)
        Returns:  OpenSet
            Either one of the original open sets, when one already represents
            the intersection, or a new open set combining the factors of both."""
        if not isinstance(other, OpenSet):
            raise ValueError("Can only intersect with another OpenSet")
        if self.manifold!=other.manifold:
            raise ValueError("Cannot intersect open sets belonging to different manifolds.")
        merged=sorted(set(self._effective_factors)|set(other._effective_factors),
                      key=lambda U: str(U.name))
        if tuple(merged)==self._effective_factors: return self
        if tuple(merged)==other._effective_factors: return other
        new_name=Str("∩".join(str(U.name) for U in merged))
        return OpenSet(new_name, self.manifold, merged)
    
class MetricManifold(Basic):
    """Represent a manifold together with a compatible metric pair."""
    def __new__(cls, manifold, covariant_metric=None, contravariant_metric=None):
        """Create a metric manifold from one or both metric representations."""
        from tensorium.riemannian.metric import ContravariantMetricTensor, CovariantMetricTensor

        if covariant_metric is None and contravariant_metric is None:
            raise ValueError("At least one of covariant_metric or contravariant_metric must be provided.")
        if covariant_metric is not None and not isinstance(covariant_metric, CovariantMetricTensor):
            raise ValueError("covariant_metric must be a CovariantMetric")
        if contravariant_metric is not None and not isinstance(contravariant_metric, ContravariantMetricTensor):
            raise ValueError("contravariant_metric must be a ContravariantMetric")
        if covariant_metric is not None and covariant_metric.manifold!=manifold:
            raise ValueError("covariant_metric must belong to the manifold.")
        if contravariant_metric is not None and contravariant_metric.manifold!=manifold:
            raise ValueError("contravariant_metric must belong to the manifold.")
        
        if covariant_metric is None:
            covariant_metric=contravariant_metric.inverse()
        if contravariant_metric is None:
            contravariant_metric=covariant_metric.inverse()

        obj=super().__new__(cls, manifold, covariant_metric, contravariant_metric)
        return obj
    
    @property
    def manifold(self):
        """Return the underlying manifold."""
        return self.args[0]
    @property
    def base_manifold(self):
        """Return the underlying manifold without metric wrapper."""
        return self.args[0]
    @property
    def dim(self):
        """Return the dimension inherited from the underlying manifold."""
        return self.base_manifold.dim
    @property
    def covariant_metric(self):
        """Return the covariant metric tensor."""
        return self.args[1]
    @property
    def contravariant_metric(self):
        """Return the contravariant metric tensor."""
        return self.args[2]

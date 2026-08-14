from sympy.core import Basic
from tensorium.core.manifold import Manifold, MetricManifold
from tensorium.core.chart import Chart

class Connection(Basic):
    """Base class for connections defined over a manifold."""
    def __new__(cls, manifold):
        """Create a connection over a manifold.
        Parameters:
         - manifold : Manifold or MetricManifold
            Manifold on which the connection is defined.
        Returns:
         - Connection
            Connection object over ``manifold``.
        """
        if not (isinstance(manifold, Manifold) or isinstance(manifold, MetricManifold)):
            raise ValueError("The connection can only be defined on a Manifold.")
        obj = super().__new__(cls, manifold)
        obj._manifold=manifold
        return obj
    @property
    def manifold(self): return self.args[0]
    @property
    def base_manifold(self):
        if isinstance(self.manifold, MetricManifold):
            return self.manifold.base_manifold
        return self.manifold
    @property
    def dim(self): return self.manifold.dim

class AffineConnection(Connection):
    """Affine connection described by local Christoffel symbols."""
    def __new__(cls, manifold, local_representations):
        """Create an affine connection from local representations.
        Parameters:
         - manifold : Manifold or MetricManifold
            Manifold on which the connection is defined.
         - local_representations : dict
            Dictionary mapping charts to ``LocalAffineConnection`` objects.
        Returns:
         - AffineConnection
            Global affine connection with the provided local data.
        """
        if not (isinstance(manifold, Manifold) or isinstance(manifold, MetricManifold)):
            raise ValueError("The connection can only be defined on a Manifold.")
        for chart, local_conn in local_representations.items():
            if not isinstance(local_conn, LocalAffineConnection):
                raise ValueError("All local representations must be LocalAffineConnection objects.")
            if local_conn.chart != chart:
                raise ValueError("Each local connection must be stored under its own chart.")
        obj = super().__new__(cls, manifold)
        obj._local_representations=local_representations
        return obj
    @property
    def local_representations(self): return self._local_representations

class LocalAffineConnection(Basic):
    """Affine connection written in a single chart."""
    def __new__(cls, chart, components):
        """Create local connection coefficients in one chart.
        Parameters:
         - chart : Chart
            Chart in which the coefficients are expressed.
         - components : object
            Components of the connection, interpreted as ``Γ^i_{jk}``.
        Returns:
         - LocalAffineConnection
            Local affine connection in ``chart``.
        """
        if not isinstance(chart, Chart):
            raise ValueError("The local connection must be defined on a chart.")
        obj=super().__new__(cls, chart, components)
        obj._chart=chart
        obj._components=components
        return obj
    @property
    def chart(self): return self.args[0]
    @property
    def components(self): return self.args[1]

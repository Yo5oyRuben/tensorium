from tensorium.connections.base import Connection
from tensorium.core.manifold import Manifold, MetricManifold
from tensorium.fields.valued_tensor_field import ValuedTensorField, nested_shape

class GaugeConnection(Connection):
    """Matrix-valued one-form acting on internal indices in a fixed trivialization.

    The connection is represented locally as an ``End(E)``-valued one-form with
    one upper and one lower internal index. This is the trivialized/local
    description used in gauge theory; general bundle transition functions are
    not modeled explicitly.
    """
    def __new__(cls,manifold,internal_tensor_field):
        """Create a gauge connection from an valued tensor field.
        Parameters:
         - manifold : Manifold or MetricManifold
            Base manifold on which the connection is defined.
         - internal_tensor_field : ValuedTensorField
            One-form-valued internal matrix with tensor type ``(0,1)`` and
            internal variance ``(1,-1)``.
        Returns:
         - GaugeConnection
            Gauge connection in a fixed internal trivialization.
        """
        if not (isinstance(manifold, Manifold) or isinstance(manifold, MetricManifold)): raise ValueError("The connection can only be defined on a Manifold.")
        if not isinstance(internal_tensor_field, ValuedTensorField): raise ValueError("The connection must be defined by a ValuedTensorField.")
        base_manifold=manifold.base_manifold if isinstance(manifold, MetricManifold) else manifold
        if internal_tensor_field.base_manifold!=base_manifold: raise ValueError("The base manifold of the valued tensor field must coincide with the manifold of the connection.")
        if internal_tensor_field.tensor_type!=(0,1): raise ValueError("gauge connection must be one-form valued")
        if internal_tensor_field.index_variance!=(-1,): raise ValueError("gauge connection must have one covariant geometric index")
        if internal_tensor_field.internal_rank!=2: raise ValueError("gauge connection must have two internal indices")
        if internal_tensor_field.internal_shape[0]!=internal_tensor_field.internal_shape[1]: raise ValueError("gauge connection must have square internal indices")
        if internal_tensor_field.internal_variance!=(1,-1): raise ValueError("gauge connection must have one covariant and one contravariant internal index")
        obj=super().__new__(cls,manifold)
        obj._internal_tensor_field=internal_tensor_field
        return obj
    
    @property
    def internal_tensor_field(self): return self._internal_tensor_field
    @property
    def internal_shape(self): return self.internal_tensor_field.internal_shape
    @property
    def internal_variance(self): return self.internal_tensor_field.internal_variance
    @property
    def internal_dim(self): return self.internal_shape[0]
    @property
    def components(self): return self.internal_tensor_field.components
    def __repr__(self):
        return f"GaugeConnection(internal_dim={self.internal_dim}, manifold={self.manifold})"

    def __getitem__(self, indices):
        """Return one internal matrix component.
        Parameters:
         - indices : tuple
            Pair of internal indices.
        Returns:
         - TensorField
            One-form component selected by ``indices``.
        """
        return self.internal_tensor_field[indices]
    def local_representation(self, chart):
        """Return the local matrix-valued one-form in a chart.
        Parameters:
         - chart : Chart
            Chart in which the components should be expressed.
        Returns:
         - tuple
            Nested tuple of local one-form components.
        """
        return self.internal_tensor_field.local_representation(chart)
    def act_on(self,field):
        """Apply the gauge connection to an internal field.
        Parameters:
         - field : ValuedTensorField
            Valued field on which the connection acts.
        Returns:
         - ValuedTensorField or TensorField
            Internal action of the matrix-valued one-form on ``field``.
        """
        return self.internal_tensor_field.internal_action(field, pairs=[(1,0)])

    @classmethod
    def from_components(cls,manifold,components):
        """Build a gauge connection from matrix-valued one-form components.
        Parameters:
         - manifold : Manifold or MetricManifold
            Base manifold on which the connection is defined.
         - components : nested list or tuple
            Components of the internal matrix-valued one-form.
        Returns:
         - GaugeConnection
            Gauge connection built from the provided components.
        """
        internal_shape=nested_shape(components)
        field=ValuedTensorField(components,internal_shape,(1,-1))
        return cls(manifold,field)

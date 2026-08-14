from sympy.core import S
from sympy.core.sympify import _sympify
from tensorium.connections import Connection, AffineConnection, LocalAffineConnection
from tensorium.fields.tensor_field import TensorField
from tensorium.fields.local_tensor_field import LocalTensorField
from tensorium.core.chart import chart_diff
from itertools import product
from sympy import cancel, factor_terms
from sympy.tensor import ImmutableDenseNDimArray
from tensorium.fields.valued_tensor_field import ValuedTensorField, nested_map, _build_nested_components
from tensorium.operators.base import TensorOperator

class CovariantDerivative(TensorOperator):
    """Covariant derivative with optional affine and trivialized gauge parts.

    The affine part acts on geometric tensor indices. The gauge part acts on
    finite-dimensional internal indices written in a fixed internal basis.
    """
    def __init__(self, affine_connection=None,gauge_connection=None):
        """Create a covariant derivative operator.
        Parameters:
         - affine_connection : AffineConnection, optional
            Connection acting on geometric tensor indices.
         - gauge_connection : Connection, optional
            Matrix-valued one-form acting on internal indices.
        Returns:
         - CovariantDerivative
            Unary tensor operator with one covariant free geometric index.
        """
        if affine_connection is None and gauge_connection is None:
            raise ValueError("At least one of affine_connection or gauge_connection must be provided.")
        if affine_connection is not None and not isinstance(affine_connection, AffineConnection):
            raise ValueError("affine_connection must be an AffineConnection")
        if gauge_connection is not None and not isinstance(gauge_connection, Connection):
            raise ValueError("gauge_connection must be a Connection")
        if affine_connection is not None and gauge_connection is not None:
            if affine_connection.base_manifold!=gauge_connection.base_manifold:
                raise ValueError("The affine and gauge connections must be defined on the same base manifold.")
        self._affine_connection=affine_connection
        self._gauge_connection=gauge_connection
        if gauge_connection is None: signature=TensorOperator._generic_unary_signature((-1,))
        else: signature=TensorOperator._generic_internal_unary_signature((-1,))
        super().__init__(index_variance=(-1,),action=self._apply,name="∇",arity=1,signature=signature,operator_kind="covariant_derivative")

    @property
    def index_variance(self): return (-1,) 
    @property
    def affine_connection(self): return self._affine_connection
    @property
    def gauge_connection(self): return self._gauge_connection
    @property
    def connections(self): return tuple(c for c in (self.affine_connection, self.gauge_connection) if c is not None)
    @property
    def _reference_connection(self):
        if self.affine_connection is not None: return self.affine_connection
        return self.gauge_connection
    @property
    def base_manifold(self): return self._reference_connection.base_manifold
    @property
    def manifold(self): return self._reference_connection.manifold
    def __repr__(self):
        parts=[]
        if self.affine_connection is not None: parts.append("affine")
        if self.gauge_connection is not None: parts.append("gauge")
        return f"CovariantDerivative({'+'.join(parts)})"

    def _apply(self,field):
        """Apply the affine/gauge covariant derivative to a field.
        Parameters:
         - field : TensorField or ValuedTensorField
            Field on which the derivative acts.
        Returns:
         - TensorField or ValuedTensorField
            Covariant derivative of ``field``.
        """
        if isinstance(field,TensorField):
            if self.gauge_connection is not None:
                raise ValueError("gauge connection can only act on fields with internal indices")
            return AffineCovariantDerivative(self.affine_connection)(field)
        if isinstance(field,ValuedTensorField):
            return self._internal_covariant_derivative(field)
        raise ValueError("unsupported field type")

    def _internal_covariant_derivative(self,field):
        """Apply the full covariant derivative to a field with internal values.
        Parameters:
         - field : ValuedTensorField
            Field carrying internal indices.
        Returns:
         - ValuedTensorField
            Sum of the affine and gauge contributions that are available.
        """
        if not isinstance(field,ValuedTensorField):
            raise ValueError("field must be a ValuedTensorField")
        if self.affine_connection is None and self.gauge_connection is None:
            raise ValueError("At least one of affine_connection or gauge_connection must be provided.")
        if self.base_manifold!=field.base_manifold:
            raise ValueError("The field and the connection must be defined on the same base manifold.")
        result=None
        if self.affine_connection is not None: result=self._affine_part_on_internal_field(field)
        if self.gauge_connection is not None:
            gauge_part=self._gauge_part_on_internal_field(field)
            result=gauge_part if result is None else result+gauge_part
        return result

    def _affine_part_on_internal_field(self,field):
        """Apply the affine part component by component.
        Parameters:
         - field : ValuedTensorField
            Valued tensor field.
        Returns:
         - ValuedTensorField
            Field obtained by applying the affine covariant derivative to each
            tensor component.
        """
        if not isinstance(field,ValuedTensorField): raise ValueError("field must be a ValuedTensorField")
        if self.affine_connection is None: raise ValueError("affine_connection must be provided for the affine part")
        return field._new_like(nested_map(lambda component: AffineCovariantDerivative(self.affine_connection)(component), field.components))

    def _gauge_part_on_internal_field(self,field):
        """Apply the gauge part to all internal indices.
        Parameters:
         - field : ValuedTensorField
            Valued tensor field.
        Returns:
         - ValuedTensorField
            Gauge contribution to the covariant derivative. Contravariant
            internal indices receive a plus sign and covariant ones a minus
            sign.
        """
        if not isinstance(field,ValuedTensorField): raise ValueError("field must be a ValuedTensorField")
        if self.gauge_connection is None: raise ValueError("gauge_connection must be provided for the gauge part")
        if field.base_manifold!=self.gauge_connection.base_manifold: raise ValueError("The field and the gauge connection must be defined on the same base manifold.")
        if any(dim != self.gauge_connection.internal_dim for dim in field.internal_shape):
            raise ValueError("all internal dimensions must match the gauge connection internal dimension")

        internal_variance=field.internal_variance
        if isinstance(internal_variance,int): internal_variance=(internal_variance,)
        dim_internal=field.internal_shape[0]
        def build_component(output_internal_index):
            total=None
            for pos in range(field.internal_rank):
                current_index=output_internal_index[pos]
                variance=internal_variance[pos]
                for dummy in range(dim_internal):
                    source_index=list(output_internal_index)
                    source_index[pos]=dummy
                    source_index=tuple(source_index)
                    if variance==1: term=self.gauge_connection[current_index,dummy].tensor_product(field[source_index])
                    else: term=-self.gauge_connection[dummy,current_index].tensor_product(field[source_index])
                    if field.rank>0:
                        order=tuple(range(1,field.rank+1))+(0,)
                        term=term.permute_indices(order)
                    total=term if total is None else total+term
            return total
        components=_build_nested_components(field.internal_shape,build_component)
        return ValuedTensorField(components,field.internal_shape,internal_variance)
        
class AffineCovariantDerivative(CovariantDerivative):
    """Covariant derivative operator associated with an affine connection."""
    def __init__(self, connection):
        """Create an affine covariant derivative operator.
        Parameters:
         - connection : AffineConnection
            Affine connection used to differentiate tensor fields.
        Returns:
         - AffineCovariantDerivative
            Unary tensor operator with one covariant free geometric index.
        """
        if not isinstance(connection, AffineConnection):
            raise ValueError("connection must be a AffineConnection")
        super().__init__(affine_connection=connection)
    
    def __call__(self, tensor):
        """Apply the affine covariant derivative to a tensor field.
        Parameters:
         - tensor : TensorField
            Tensor field to differentiate.
        Returns:
         - TensorField
            Covariant derivative of ``tensor``.
        """
        if not isinstance(tensor, TensorField):
            raise ValueError("The covariant derivative operator must act on Tensor Fields")
        if self.affine_connection.base_manifold!=tensor.base_manifold:
            raise ValueError("The Tensor Field and the Connection are not defined on the same manifold")
        common_charts=set(tensor.local_representations).intersection(self.affine_connection.local_representations)
        representations={}
        if common_charts:
            for chart in common_charts:
                representations[chart]=local_covariant_derivative(tensor.local_representation(chart),self.affine_connection.local_representation(chart))
        else:
            candidate_charts=set(tensor.local_representations)|set(self.affine_connection.local_representations)
            target_chart=min(candidate_charts,key=lambda c:distance_to_chart(tensor,c)+distance_to_chart(self.affine_connection,c))
            representations[target_chart]=local_covariant_derivative(tensor.local_representation(target_chart), self.affine_connection.local_representation(target_chart))
        return TensorField(tensor.manifold, (tensor.contravariant_order,tensor.covariant_order+1),
                           representations,tensor.index_variance+(-1,))
    
def local_covariant_derivative(local_tensor, local_connection):
    """Compute the local covariant derivative of a tensor field.
    Parameters:
     - local_tensor : LocalTensorField
        Tensor field represented in a single chart.
     - local_connection : LocalAffineConnection
        Local affine connection in the same chart.
    Returns:
     - LocalTensorField
        Local tensor field representing the covariant derivative.
    """
    if not isinstance(local_connection, LocalAffineConnection):
        raise ValueError("local_connection must be a LocalAffineConnection")
    if not isinstance(local_tensor, LocalTensorField):
        raise ValueError("local_tensor must be a LocalTensorField")
    if local_connection.chart!=local_tensor.chart:
        raise ValueError("local_tensor and local_connection must belong to the same chart")
    tensor_get=local_tensor.__getitem__
    conn_get=local_connection.__getitem__
    r=local_tensor.contravariant_order
    s=local_tensor.covariant_order
    rank=r+s
    dim=local_tensor.chart.dim
    index_variance=local_tensor.index_variance

    data=[]
    for exterior_index in product(range(dim), repeat=rank+1):
        base_index=exterior_index[:-1]
        mu=exterior_index[-1]
        total=chart_diff(_sympify(tensor_get(base_index)), local_tensor.chart, mu)
        for pos in range(rank):
            current=base_index[pos]
            if index_variance[pos]==1:
                correction=S.Zero
                for nu in range(dim):
                    replaced=list(base_index)
                    replaced[pos]=nu
                    correction+=conn_get((current,mu,nu))*tensor_get(tuple(replaced))
                total+=correction
            else:
                correction=S.Zero
                for nu in range(dim):
                    replaced=list(base_index)
                    replaced[pos]=nu
                    correction+=conn_get((nu,mu,current))*tensor_get(tuple(replaced))
                total-=correction
        data.append(factor_terms(cancel(total)))
    return LocalTensorField(local_tensor.chart, (r,s+1), ImmutableDenseNDimArray(data, (dim,)*(rank+1)),
                            local_tensor.index_variance+(-1,))

def distance_to_chart(obj, chart):
    """Return the minimal atlas distance from an object to a target chart.
    Parameters:
     - obj : TensorField or AffineConnection
        Object with known local representations.
     - chart : Chart
        Target chart.
    Returns:
     - int
        Minimal number of transition maps needed to reach ``chart``.
    """
    atlas=obj.manifold.atlas
    return min(len(atlas.find_path(source_chart, chart))-1 for source_chart in obj.local_representations)

def covariant_derivative(field,affine_connection=None,gauge_connection=None):
    """Apply a covariant derivative without explicitly storing the operator.
    Parameters:
     - field : TensorField or ValuedTensorField
        Field to differentiate.
     - affine_connection : AffineConnection, optional
        Connection acting on geometric tensor indices.
     - gauge_connection : Connection, optional
        Connection acting on internal indices.
    Returns:
     - TensorField or ValuedTensorField
        Covariant derivative of ``field``.
    """
    return CovariantDerivative(affine_connection=affine_connection,gauge_connection=gauge_connection)(field)

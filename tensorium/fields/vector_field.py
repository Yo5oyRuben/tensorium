from tensorium.fields.tensor_field import TensorField
from tensorium.fields.local_tensor_field import LocalTensorField
from tensorium.core.chart import chart_diff
from sympy.core import S
from sympy.tensor import ImmutableDenseNDimArray

class Vector(TensorField):
    """Represent a vector field on a manifold.
    A ``Vector`` is a specialized ``TensorField`` of type ``(1, 0)``.
    Its known local representations are stored as local tensor fields on
    specific charts, and additional representations can be derived through
    chart transformations.
    """
    def __init__(self, manifold, local_representations):
        """Create a vector field from local representations.
        Parameters:
         - manifold : Manifold (Manifold on which the vector field is defined)
         - local_representations : dict
            Dictionary mapping charts to local tensor representations of the
            vector field.
        Notes:
        A vector field is always treated as a tensor of type ``(1, 0)`` with
        index variance ``(1,)``.
        """
        super().__init__(manifold, (1,0), local_representations, (1,))
    
    def __call__(self, *args):
        """Apply the vector field to a scalar field or a one-form.
        Parameters:
         - *args : TensorField
            Exactly one argument must be provided.
            - If the argument is a scalar field of type ``(0, 0)``, the vector
            acts by directional differentiation.
            - If the argument is a one-form of type ``(0, 1)``, the evaluation
            is delegated to the generic tensor contraction logic.
        Returns
         - TensorField: A rank-zero tensor field representing the resulting scalar field.
        Notes:
        When acting on a scalar field, the method selects a suitable common
        chart, expresses both objects in that chart, and computes the usual
        coordinate formula
        ``X(f) = sum_i X^i d f / d x^i``.
        """
        if(len(args)!=1):
            raise ValueError("A vector field must act on either scalar fields by differentiation, or on 1-form fields."
                             "Thus, it must act on exactly one argument")
        arg=args[0]
        if isinstance(arg, TensorField):
            if arg.tensor_type!=(0,0): raise ValueError("The TensorField supplied is not a scalar field: (0,0) TensorField")
        else: raise ValueError("The argument supplied is not a TensorField. It should be either a scalar field ((0,0)-TensorField)" \
        "or a OneForm field ((0,1)-TensorField)")
        if arg.tensor_type==(0,1):
            return super().__call__(arg)

        common_charts=set(self.local_representations).intersection(arg.local_representations)
        representations={}
        if common_charts:
            target_charts=common_charts
        else:
            target_charts={self._best_call_chart((arg,))}

        for target_chart in target_charts:
            local_vector=self.local_representation(target_chart)
            local_scalar=arg.local_representation(target_chart)
            total=S.Zero
            scalar_expr=local_scalar.components[()]
            for i in range(target_chart.dim):
                total+=local_vector.components[(i,)]*chart_diff(scalar_expr, target_chart, i)
            representations[target_chart]=LocalTensorField(target_chart, (0,0), total, ())
        return TensorField(self.manifold, (0,0), representations, ())
    
    def commutator(self, other):
        """Return the Lie bracket of two vector fields.
        Parameters:
         - other : Vector (Vector field to bracket with ``self``)
        Returns:
         - Vector: The commutator ``[self, other]`` as a new vector field.
        Notes:
        If the vector fields already share a chart, the bracket is computed
        there. Otherwise, both are first transformed to a suitable common chart.
        """
        if not isinstance(other, Vector):
            raise ValueError("One of the arguments supplied is not a vector")
        if other.manifold!=self.manifold: raise ValueError("The two Vector fields are not defined on the same manifold")
        common_charts=set(self.local_representations).intersection(other.local_representations)
        representations={}
        if common_charts:
            for chart in common_charts:
                representations[chart]=local_commutator(self.local_representation(chart),other.local_representation(chart))
        else:
            target_chart=self._best_call_chart((other,))
            other_transformed=other.local_representation(target_chart)
            self_transformed=self.local_representation(target_chart)
            representations[target_chart]=local_commutator(self_transformed, other_transformed)
        return Vector(self.manifold, representations)
    
def local_commutator(local_X, local_Y):
    """Compute the commutator of two local vector fields in one chart.
    Parameters:
     - local_X : LocalTensorField (Local representation of the first vector field)
     - local_Y : LocalTensorField (Local representation of the second vector field)
    Returns:
     - LocalTensorField: Local vector field representing the commutator ``[X, Y]``.
    Notes:
    This function implements the coordinate expression
    ``[X, Y]^i = X^j ∂_j Y^i - Y^j ∂_j X^i``.
    It assumes both local vector fields are already expressed in the same
    chart.
    """
    data=[]
    dim=local_X.chart.dim
    for i in range(dim):
        total=S.Zero
        total+=sum(local_X.components[(j,)]*chart_diff(local_Y.components[(i,)], local_X.chart, j) for j in range(dim))
        total-=sum(local_Y.components[(j,)]*chart_diff(local_X.components[(i,)], local_X.chart, j) for j in range(dim))
        data.append(total)
    return LocalTensorField(local_X.chart,(1,0), ImmutableDenseNDimArray(data, (dim,)))

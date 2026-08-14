from tensorium.fields.tensor_field import TensorField
from tensorium.fields.local_tensor_field import LocalTensorField
import symengine as se
from tensorium.utils.conversions import _to_symengine, _to_sympy

#__init__(self, manifold, tensor_type, local_representations, index_variance=None):
class LocalCovariantMetricTensor(LocalTensorField):
    """Represent a covariant metric tensor in a single chart."""
    def __new__(cls, chart, components):
        """Create a symmetric local covariant metric tensor."""
        dim=chart.dim
        equivalences = {(i,j):((j,i),1) for i in range(dim) for j in range(dim) if i>j}
        return super().__new__(cls,chart, (0,2), components, (-1,-1),equivalences)
    def inverse(self):
        """Return the inverse contravariant metric in the same chart."""
        dim=self.chart.dim
        g_mat=self.to_matrix()
        locals_dict={}
        for entry in g_mat:
            for sym in entry.free_symbols: locals_dict.setdefault(str(sym), sym)
        se_mat=se.Matrix(dim, dim, [_to_symengine(g_mat[i,j]) for i in range(dim) for j in range(dim)])
        se_inv=se_mat.inv()

        data=tuple(_to_sympy(se_inv[i,j], local_dict=locals_dict) for i in range(dim) for j in range(i,dim))
        return LocalContravariantMetricTensor(self.chart, data)
    
class LocalContravariantMetricTensor(LocalTensorField):
    """Represent a contravariant metric tensor in a single chart."""
    def __new__(cls, chart, components):
        """Create a symmetric local contravariant metric tensor."""
        dim=chart.dim
        equivalences = {(i,j):((j,i),1) for i in range(dim) for j in range(dim) if i>j}
        return super().__new__(cls,chart, (2,0), components, (1,1),equivalences)
    def inverse(self):
        """Return the inverse covariant metric in the same chart."""
        dim=self.chart.dim
        g_mat=self.to_matrix()
        locals_dict={}
        for entry in g_mat:
            for sym in entry.free_symbols: locals_dict.setdefault(str(sym), sym)
        se_mat=se.Matrix(dim, dim, [_to_symengine(g_mat[i,j]) for i in range(dim) for j in range(dim)])
        se_inv=se_mat.inv()

        data=tuple(_to_sympy(se_inv[i,j], local_dict=locals_dict) for i in range(dim) for j in range(i,dim))
        return LocalCovariantMetricTensor(self.chart, data)

class CovariantMetricTensor(TensorField):
    """Represent a covariant metric tensor field on a manifold."""
    def __init__(self, manifold, local_representations):
        """Create a covariant metric tensor field from local metrics."""
        for local_metric in local_representations.values():
            if not isinstance(local_metric, LocalCovariantMetricTensor):
                raise ValueError("All local representations must be LocalCovariantMetricTensor objects.")
        super().__init__(manifold, (0,2), local_representations, (-1,-1))

    def inverse(self):
        """Return the associated contravariant metric tensor field."""
        inverse_locals={chart: local_metric.inverse() for chart, local_metric in self.local_representations.items()}
        return ContravariantMetricTensor(self.manifold, inverse_locals)
    
class ContravariantMetricTensor(TensorField):
    """Represent a contravariant metric tensor field on a manifold."""
    def __init__(self, manifold, local_representations):
        """Create a contravariant metric tensor field from local metrics."""
        for local_metric in local_representations.values():
            if not isinstance(local_metric, LocalContravariantMetricTensor):
                raise ValueError("All local representations must be LocalContravariantMetricTensor objects.")
        super().__init__(manifold, (2,0), local_representations, (1,1))

    def inverse(self):
        """Return the associated covariant metric tensor field."""
        inverse_locals={chart: local_metric.inverse() for chart, local_metric in self.local_representations.items()}
        return CovariantMetricTensor(self.manifold, inverse_locals)

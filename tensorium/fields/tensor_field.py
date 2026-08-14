from sympy.core import S

from tensorium.core import atlas
from tensorium.core.manifold import MetricManifold, Manifold

class TensorField():
    """Represent a tensor field on a manifold through its local coordinate representations.
    A ``TensorField`` stores one or more known local representations, each
    given as a ``LocalTensorField`` on a specific chart. Additional local
    representations can be computed on demand by transforming from a known
    chart through the manifold atlas.
    The tensor type is described by a pair ``(r, s)``, where ``r`` is the
    contravariant order and ``s`` is the covariant order.
    """
    def __init__(self, manifold, tensor_type, local_representations, index_variance=None):
        """Create a tensor field from known local representations.
        Parameters:
         - manifold : Manifold (Manifold on which the tensor field is defined)
         - tensor_type : tuple
            Pair ``(r, s)`` describing the contravariant and covariant orders
            of the tensor.
         - local_representations : dict
            Dictionary mapping charts to ``LocalTensorField`` objects that
            represent the tensor in those charts.
         - index_variance : tuple, optional
            Variance pattern of the tensor indices. Use ``1`` for
            contravariant positions and ``-1`` for covariant positions. If
            omitted, the default is all contravariant indices first, followed
            by all covariant indices.
        """
        if not (isinstance(manifold, Manifold) or isinstance(manifold, MetricManifold)):
            raise ValueError("manifold must be a Manifold (or a MetricManifold)")
        if not isinstance(local_representations, dict): raise ValueError("local_representations must be a dictionary")
        if index_variance is None:
            index_variance=(1,)*tensor_type[0]+(-1,)*tensor_type[1]
        first_tensor_index_variance=next(iter(local_representations.values())).index_variance
        for chart, local_tensor_field in local_representations.items():
            if isinstance(manifold, MetricManifold):
                if chart.manifold!=manifold.manifold: raise ValueError("Some chart in the local_representations" \
            "dictionary does not belong to the same Manifold as manifold")
            else:
                if chart.manifold!=manifold: raise ValueError("Some chart in the local_representations" \
            "dictionary does not belong to the same Manifold as manifold")
            if first_tensor_index_variance!=local_tensor_field.index_variance:
                raise ValueError("Not all LocalTensorField in local_representations are of the same type")
        self._manifold=manifold
        self._tensor_type=tensor_type
        self._index_variance=index_variance
        self._local_representations=local_representations
    
    @property
    def manifold(self):
        """Return the manifold on which the tensor field is defined."""
        return self._manifold
    @property
    def base_manifold(self):
        """Return the underlying manifold, unwrapping MetricManifold when needed."""
        if isinstance(self._manifold, MetricManifold): return self._manifold.manifold
        return self._manifold
    @property
    def tensor_type(self):
        """Return the tensor type ``(r, s)``."""
        return self._tensor_type
    @property
    def index_variance(self):
        """Return the variance pattern of the tensor indices."""
        return self._index_variance
    @property
    def local_representations(self):
        """Return the known local tensor representations."""
        return self._local_representations
    @property
    def contravariant_order(self):
        """Return the number of contravariant indices."""
        return self.tensor_type[0]
    @property
    def covariant_order(self):
        """Return the number of covariant indices."""
        return self.tensor_type[1]
    @property
    def rank(self):
        """Return the total tensor rank."""
        return self.covariant_order+self.contravariant_order
    @property
    def signature(self):
        """Return the field signature associated with this tensor field.
        Returns:
         - FieldSignature
            Signature containing tensor type, index variance and base manifold.
        """
        from tensorium.operators.signature import FieldSignature
        return FieldSignature(tensor_type=self.tensor_type,index_variance=self.index_variance,manifold=self.base_manifold)

    def _best_source_chart(self, target_chart):
        """Select the known source chart that is closest to a target chart.
        Parameters:
         - target_chart : Chart
            Chart in which a local representation is requested.
        Returns:
         - Chart or None:
            Known chart from which the transformation to ``target_chart`` uses
            the shortest atlas path, or ``None`` if no suitable chart is found.
        """
        best_chart=None
        best_length=None
        for chart in self.local_representations:
            path=target_chart.manifold.atlas.find_path(chart, target_chart)
            if best_length is None or len(path)<best_length:
                best_length=len(path)
                best_chart=chart
        return best_chart
    
    def local_representation(self, target_chart):
        """Return a local representation of the tensor field in a target chart.
        Parameters:
         - target_chart : Chart
            Chart in which the tensor field should be represented.
        Returns:
         - LocalTensorField
            Representation of the tensor field in ``target_chart``.
        Notes
        If the representation is not already known, the method transforms a
        known local representation to the target coordinates. The transformed
        representation is defined on the overlap of the source and target chart
        domains.
        """
        if target_chart in self.local_representations:
            return self.local_representations[target_chart]
        
        source_chart=self._best_source_chart(target_chart)
        if source_chart is None:
            raise KeyError("No known local representation is connected to the target chart")
        source_local_representation=self.local_representations[source_chart]
        target_local_representation=source_local_representation.transform_to(target_chart)

        self.local_representations[target_local_representation.chart]=target_local_representation
        return target_local_representation
    
    def _best_call_chart(self, args):
        """Choose the most convenient chart for tensor evaluation.
        Parameters:
         - args : iterable of TensorField
            Tensor field arguments that will be fed into this tensor.
        Returns
         - Chart:
            Chart minimizing the total transformation cost needed to express
            both this tensor field and all arguments in a common chart.
        """
        best_chart=None
        best_cost=None
        for target_chart in self.local_representations:
            total_cost=0
            possible=True
            for arg in args:
                min_cost_for_arg=None
                for source_chart in arg.local_representations:
                    try:
                        path=self.manifold.atlas.find_path(source_chart, target_chart)
                        cost=len(path)-1
                        if min_cost_for_arg is None or cost<min_cost_for_arg:
                            min_cost_for_arg=cost
                    except KeyError: pass
                if min_cost_for_arg is None:
                    possible=False
                    break
                total_cost+=min_cost_for_arg
            if possible and (best_cost is None or total_cost<best_cost):
                best_cost=total_cost
                best_chart=target_chart
        if best_chart is None: raise KeyError("No common reachable chart was found for the tensor field and all arguments.")
        return best_chart
    
    def __call__(self, *args):
        """Evaluate the tensor field on tensor arguments of rank one.
        Parameters:
         - *args : TensorField
            Arguments to be inserted into the tensor slots. Each argument must
            be a tensor field of rank one:
            - a ``(0, 1)`` tensor for contravariant slots,
            - a ``(1, 0)`` tensor for covariant slots.
        Returns:
         - TensorField: A rank-zero tensor field representing the resulting scalar field.
        Notes:
        The method first selects a common chart that minimizes transformation
        cost, converts all arguments to that chart, and then delegates the local
        evaluation to the corresponding ``LocalTensorField`` objects.
        """
        r=self.contravariant_order
        s=self.covariant_order
        rank=r+s

        if(rank==0): return self
        if rank!=len(args):
            raise ValueError("The number of arguments does not match the rank of the tensor field")
        for i in range(rank):
            if not isinstance(args[i], TensorField): raise ValueError("One of the arguments is not a Tensor Field"
            "(neither a Vector field nor a OneForm field)")
            if self.index_variance[i]==1 and args[i].tensor_type!=(0,1):
                 raise ValueError(f"Argument {i} must be a OneForm, since index {i} is contravariant.")
            if self.index_variance[i]==-1 and args[i].tensor_type!=(1,0):
                raise ValueError(f"Argument {i} must be a Vector, since index {i} is covariant.")

        common_charts=set(self.local_representations)
        for arg in args:
            common_charts=common_charts.intersection(arg.local_representations)

        representations={}
        if common_charts:
            for chart in common_charts:
                args_coord=[arg.local_representations[chart] for arg in args]
                representations[chart]=self.local_representations[chart](*args_coord)
        else:
            target_chart=self._best_call_chart(args)
            args_coord=[arg.local_representation(target_chart) for arg in args]
            representations[target_chart]=self.local_representation(target_chart)(*args_coord)
        return TensorField(self.manifold, (0,0), representations, ())
    
    def __add__(self, other):
        """Add two tensor fields of the same type on the same manifold.
        Parameters:
         - other : TensorField
            Tensor field to add to ``self``.
        Returns:
         - TensorField: Tensor field representing the sum.
        Notes:
        If the two tensor fields already share one or more charts, the sum is
        computed on each common chart. Otherwise, both tensors are transformed
        to a suitable common chart before adding their local representations.
        """
        if not isinstance(other, TensorField): raise ValueError("other must be a TensorField")
        if other.manifold!=self.manifold: raise ValueError("The two TensorFields are not defined on the same manifold")
        if self.tensor_type!=other.tensor_type: raise ValueError("Two tensors of different type (r,s) can not be added")
        if self.index_variance!=other.index_variance: raise ValueError("Two tensors with different index positions can not be added")
        common_charts=set(self.local_representations).intersection(other.local_representations)
        representations={}
        if common_charts:
            for chart in common_charts:
                representations[chart]=self.local_representation(chart).__add__(other.local_representation(chart))
        else:
            target_chart=self._best_call_chart((other,))
            other_transformed=other.local_representation(target_chart)
            self_transformed=self.local_representation(target_chart)
            representations[target_chart]=self_transformed.__add__(other_transformed)
        return TensorField(self.manifold, self.tensor_type, representations, self.index_variance)

    def raise_lower_indices(self, indices):
        """Raise or lower the selected indices using the manifold metric."""
        if not isinstance(self.manifold, MetricManifold):
            raise ValueError("The tensor must be defined on a MetricManifold")
        if any(i < 0 or i >= self.rank for i in indices):
            raise ValueError("Some index is out of range.")
        if len(set(indices)) != len(indices):
            raise ValueError("Repeated indices are not allowed.")
        
        tensor_charts=set(self.local_representations)
        cov_charts=set(self.manifold.covariant_metric.local_representations)
        contra_charts=set(self.manifold.contravariant_metric.local_representations)
        common_charts=tensor_charts&cov_charts&contra_charts
        res_representations={}

        indices=sorted(indices)
        new_index_variance=list(self.index_variance)
        for i in indices: new_index_variance[i]*=-1
        new_index_variance=tuple(new_index_variance)

        if common_charts:
            for chart in common_charts:
                g_cov=self.manifold.covariant_metric.local_representations[chart]
                g_contra=self.manifold.contravariant_metric.local_representations[chart]
                res_representations[chart]=self.local_representations[chart].raise_lower_indices(indices, g_cov, g_contra)
        else:
            candidate_charts=(tensor_charts|cov_charts|contra_charts)
            target_chart=min(candidate_charts, key=lambda c:distance_to_chart(self,c)+
                             distance_to_chart(self.manifold.covariant_metric,c)+
                             distance_to_chart(self.manifold.contravariant_metric,c))
            g_cov=self.manifold.covariant_metric.local_representation(target_chart)
            g_contra=self.manifold.contravariant_metric.local_representation(target_chart)
            res_representations[target_chart]=self.local_representation(target_chart).raise_lower_indices(indices,g_cov,g_contra)
        r_new=sum(1 for v in new_index_variance if v==1)
        s_new=self.rank-r_new
        return TensorField(self.manifold, (r_new, s_new), res_representations, new_index_variance)
    
    def tensor_product(self, other):
        """Return the tensor product with another tensor field."""
        if not isinstance(other, TensorField): raise ValueError("other is not a TensorField")
        if self.base_manifold!=other.base_manifold: raise ValueError("The tensors are not defined on the same manifold")
        
        common_charts=set(self.local_representations).intersection(other.local_representations)
        representations={}

        if common_charts:
            for chart in common_charts:
                representations[chart]=self.local_representations[chart].tensor_product(other.local_representations[chart])
        else:
            target_chart=self._best_call_chart((other,))
            representations[target_chart]=self.local_representation(target_chart).tensor_product(other.local_representation(target_chart))
        new_type=(self.contravariant_order+other.contravariant_order,self.covariant_order+other.covariant_order)
        
        return TensorField(self.manifold, new_type,representations,self.index_variance+other.index_variance)

    def __neg__(self):
        return TensorField(self.manifold, self.tensor_type, {chart:-local for chart, local in self.local_representations.items()}, self.index_variance)
    def __sub__(self,other): return self+(-other)
    def __mul__(self, other):
        if isinstance(other,TensorField): return self.tensor_product(other)
        return TensorField(self.manifold,self.tensor_type,{chart: local*other for chart,local in self.local_representations.items()},self.index_variance,)
    def __rmul__(self,other):
        if isinstance(other,TensorField): return other.tensor_product(self)
        return self*other

    def permute_indices(self,order):
        order=tuple(order)
        if len(order)!=self.rank: raise ValueError("order must have one entry for each tensor index")
        if tuple(sorted(order))!=tuple(range(self.rank)): raise ValueError("order must be a permutation of the indices")
        representations={chart: local.permute_indices(order) for chart,local in self.local_representations.items()}
        new_index_variance=tuple(self.index_variance[i] for i in order)
        return TensorField(self.manifold, (new_index_variance.count(1), new_index_variance.count(-1)), representations, new_index_variance)

    def contraction(self,contravariant_index,covariant_index):
        if contravariant_index<0 or contravariant_index>=self.rank: raise ValueError("contravariant_index out of range")
        if covariant_index<0 or covariant_index>=self.rank: raise ValueError("covariant_index out of range")
        if self.index_variance[contravariant_index]!=1: raise ValueError("contravariant_index must be a contravariant index")
        if self.index_variance[covariant_index]!=-1: raise ValueError("covariant_index must be a covariant index")

        representations={chart: local.contraction(contravariant_index, covariant_index) for chart, local in self.local_representations.items()}
        new_index_variance = tuple(v for i, v in enumerate(self.index_variance) if i not in (contravariant_index, covariant_index))
        return TensorField(self.manifold,(new_index_variance.count(1), new_index_variance.count(-1)),representations,new_index_variance,)                                           

def distance_to_chart(tensor, chart):
    """Return the minimal atlas distance from a tensor to a target chart."""
    atlas=tensor.manifold.atlas
    return min(len(atlas.find_path(source_chart, chart))-1 for source_chart in tensor.local_representations)

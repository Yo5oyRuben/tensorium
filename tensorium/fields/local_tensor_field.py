from sympy import cancel, factor_terms
from sympy.core import Expr, Tuple, Mul, S
from sympy.core.sympify import _sympify
from sympy.tensor import ImmutableDenseNDimArray
from sympy.matrices import ImmutableDenseMatrix
from itertools import product

from tensorium.core.manifold import OpenSet, MetricManifold
from tensorium.core.chart import Chart
from sympy.core.symbol import Str


class LocalTensorField(Expr):
    """Represent a tensor field written in a specific local chart.
    A ``LocalTensorField`` stores the components of a tensor relative to a
    single coordinate chart. Its type is given by a pair ``(r, s)``, where
    ``r`` is the number of contravariant indices and ``s`` is the number of
    covariant indices.
    The object may store either the full component array or a reduced set of
    canonical components together with symmetry relations encoded in
    ``equivalences``.
    """
    is_commutative=False

    def __new__(cls, chart, tensor_type, components, index_variance=None, equivalences=None):
        """Create a local tensor field on a chart.
        Parameters:
         - chart : Chart (Chart in which the tensor components are expressed)
         - tensor_type : tuple
            Pair ``(r, s)`` giving the contravariant and covariant orders of
            the tensor.
         - components : iterable, scalar, or ImmutableDenseNDimArray
            Tensor components in the given chart. For rank-zero tensors this may
            be a single scalar expression. For higher-rank tensors, the shape is
            expected to be ``(chart.dim,)*(r+s)`` unless ``equivalences`` is
            used.
         - index_variance : tuple, optional
            Tuple describing the variance of each index position. Use ``1`` for
            contravariant indices and ``-1`` for covariant ones. If omitted, the
            default convention is all contravariant indices first, followed by
            all covariant indices.
         - equivalences : dict, optional
            Symmetry rules relating non-canonical index tuples to canonical
            ones. Each entry should map an index tuple to a pair
            ``(canonical_index, sign)``. Example: ``{(1,0): ((0,1), -1)}`` 
            indicates that the component at index ``(1,0)`` is equal to ``-1`` 
            times the component at index ``(0,1)``.
        Returns:
         - LocalTensorField (Tensor field represented in the given chart)
        """
        if not isinstance(chart, Chart): raise ValueError("chart must be a Chart")
        if(len(tensor_type)!=2):
            raise ValueError("tensor type must be a pair (r,s)")
        
        r=_sympify(tensor_type[0])
        s=_sympify(tensor_type[1])

        if not (r.is_integer and s.is_integer and r>=0 and s>=0):
            raise ValueError("r, s must be nonnegative integers")

        if index_variance is None:
            index_variance=(1,)*r+(-1,)*s

        if equivalences is None:
            if (r+s)==0: components=ImmutableDenseNDimArray([components],())
            else:
                components=ImmutableDenseNDimArray(components)
                expected_shape=(chart.dim,)*int(r+s)
                if components.shape!=expected_shape:
                    raise ValueError(f"Expected shape {expected_shape}, got {components.shape}.") 
            return super().__new__(cls, chart,Tuple(r,s), index_variance, components, None, None)
        else:
            canonical_indices=sorted(index for index in list(product(range(chart.dim), repeat=(r+s)))
                                     if index not in equivalences
                                     )
            canonical_positions={index:pos for pos, index in enumerate(canonical_indices)}
            if len(components)!=len(canonical_indices):
                raise ValueError("The length of the components array doesn't match the expected length considering the "
                "symmetries provided")
        return super().__new__(cls, chart,Tuple(r,s), index_variance, tuple(components), equivalences, canonical_positions)
    
    @property
    def chart(self):
        """Return the chart in which the tensor is expressed."""
        return self.args[0]
    @property
    def tensor_type(self):
        """Return the tensor type ``(r, s)``."""
        return self.args[1]
    @property
    def components(self):
        """Return the stored tensor components."""
        return self.args[3]
    @property
    def covariant_order(self):
        """Return the number of covariant indices."""
        return self.tensor_type[1]
    @property
    def contravariant_order(self):
        """Return the number of contravariant indices."""
        return self.tensor_type[0]
    @property
    def rank(self):
        """Return the total tensor rank."""
        return self.contravariant_order + self.covariant_order
    @property
    def equivalences(self):
        """Return the index symmetry equivalences, if any."""
        return self.args[4]
    @property
    def canonical_positions(self):
        """Return the positions of canonical components."""
        return self.args[5]
    @property
    def index_variance(self):
        """Return the variance pattern of the tensor indices."""
        return self.args[2]
    @property
    def manifold(self):
        """Return the manifold of the underlying chart."""
        return self.chart.manifold
    @property
    def open_set(self):
        """Return the open set of the underlying chart."""
        return self.chart.open_set

    def restrict(self, W):
        """Restrict the tensor field to an open subset.
        Parameters:
         - W : OpenSet
            Open set to which the tensor field should be restricted. It must
            belong to the same manifold.
        Returns:
         - LocalTensorField (A new tensor field defined on the restricted chart)
        """
        if not isinstance(W, OpenSet): raise ValueError("W must be an OpenSet")
        if W.manifold!=self.manifold: raise ValueError("W must belong to the same manifold as the LocalTensorField")
        restricted_chart=self.chart.restrict(W)
        if restricted_chart==self.chart: return self
        subs=dict(zip(self.chart.symbols, restricted_chart.symbols))
        subs.update(dict(zip(self.chart.args[2], restricted_chart.symbols)))
        if self.rank==0:
            components=_sympify(self.components[()]).xreplace(subs)
        else:
            data=[_sympify(expr).xreplace(subs) for expr in self.components]
            if self.equivalences is None: components=ImmutableDenseNDimArray(data, self.components.shape)
            else: components=tuple(data)
        return LocalTensorField(restricted_chart, self.tensor_type, components, self.index_variance, self.equivalences)

    def __getitem__(self, indices):
        """Return the component associated with a given multi-index.
        Parameters:
         - indices : tuple
            Tuple of index values with length equal to the tensor rank.
        Returns:
         - Expr
            The requested component. If symmetry relations are present, the
            value is reconstructed from the canonical component list.
        """
        if len(indices)!=self.rank:
            raise ValueError(f"Expected {self.rank} indices, got {len(indices)}.")
        if self.equivalences is None:
            return self.components[indices]
        elif indices in self.equivalences:
            canonical_index, sign=self.equivalences[indices]
            return sign*self.components[self.canonical_positions[canonical_index]]
        else:
            return self.components[self.canonical_positions[indices]]
            
    def __add__(self, T):
        """Add two local tensor fields of the same type.
        Parameters:
         - T : LocalTensorField
            Tensor field to add. If it is expressed in a different chart, it is
            first transformed to the chart of ``self``.
        Returns:
         - LocalTensorField
            Sum of the two tensor fields, expressed in the chart of ``self``.
        Notes:
        Two tensors may have the same ``(r, s)`` type but still be incompatible
        for addition if their contravariant and covariant positions are ordered
        differently.
        """
        if not isinstance(T,LocalTensorField):
                return NotImplemented
        if self.index_variance!=T.index_variance:
            if self.contravariant_order!=T.contravariant_order or self.covariant_order!=T.covariant_order:
                raise ValueError(
                    f"The two tensor fields are not of the same type: "
                    f"({self.contravariant_order}, {self.covariant_order}) vs "
                    f"({T.contravariant_order}, {T.covariant_order})."
                )
            else:
                raise ValueError("The two tensors are of the same type (r,s). However, the position of the "
                                 "covariant and contravariant indices do not match. For example, you cann't add"
                                 r"T^{\mu}_{\nu}+S_{\mu}^{\nu}")
        if self.chart!=T.chart:
            T_coord=T.transform_to(self.chart)
        else:
            T_coord=T
        dim=self.chart.dim
        r=self.contravariant_order
        s=self.covariant_order
        if self.equivalences==T_coord.equivalences:
            data = [factor_terms(cancel(a+b)) for a,b in zip(self.components,T_coord.components)]
            if self.equivalences is None:
                components=ImmutableDenseNDimArray(data, (dim,)*(r+s))
            else:
                components=tuple(data)
            return LocalTensorField(self.chart, self.tensor_type, components, self.index_variance, self.equivalences)
        else:
            data=[]
            for multi_index in product(range(dim), repeat=(r+s)):
                data.append(factor_terms(cancel(self[multi_index]+T_coord[multi_index])))
            return LocalTensorField(self.chart, self.tensor_type, ImmutableDenseNDimArray(data, (dim,)*(r+s)), self.index_variance, None)
    
    def __mul__(self, other):
        """Multiply the tensor field by a scalar.
        Parameters:
         - other : LocalTensorField or sympifiable object
            Scalar factor. If ``other`` is a rank-zero ``LocalTensorField``, it
            is interpreted as a scalar field and transformed to the current
            chart if needed.
        Returns:
         - LocalTensorField: Tensor field scaled by ``other``.
        Notes
        Tensor-by-tensor multiplication is only supported here when the second
        factor is a scalar field. Higher-rank tensor products should be handled
        with ``tensor_product``.
        """
        if isinstance(other, LocalTensorField):
            if other.tensor_type!=(0,0): return NotImplemented
            scalar=other.transform_to(self.chart).components[()]
        else: scalar=_sympify(other)
        data=[factor_terms(a*scalar) for a in self.components]
        dim=self.chart.dim
        r=self.contravariant_order
        s=self.covariant_order
        if self.equivalences is None:
            components=ImmutableDenseNDimArray(data, (dim,)*(r+s))
        else:
            components=tuple(data)
        return LocalTensorField(self.chart, self.tensor_type, components, self.index_variance, self.equivalences)

    def __rmul__(self, other):
        """Multiply the tensor field by a scalar from the left.
        Parameters:
         - other : LocalTensorField or sympifiable object
            Scalar factor placed on the left-hand side.
        Returns:
         - LocalTensorField: Tensor field scaled by ``other``.
        """
        if isinstance(other, LocalTensorField):
            if other.tensor_type!=(0,0): return NotImplemented
            scalar=other.transform_to(self.chart).components[()]
        else: scalar=_sympify(other)
        data=[factor_terms(a*scalar) for a in self.components]
        dim=self.chart.dim
        r=self.contravariant_order
        s=self.covariant_order
        if self.equivalences is None:
            components=ImmutableDenseNDimArray(data, (dim,)*(r+s))
        else:
            components=tuple(data)
        return LocalTensorField(self.chart, self.tensor_type, components, self.index_variance, self.equivalences)
    
    def __neg__(self):
        """Return the additive inverse of the local tensor field."""
        return (-1)*self

    def __sub__(self, other):
        """Subtract another local tensor field."""
        return self+(-other)
    
    def tensor_product(self, other):
        """Return the tensor product of two local tensor fields.
        Parameters:
         - other : LocalTensorField (Tensor field to combine with ``self``)
        Returns
         - LocalTensorField
            Tensor product ``self ⊗ other`` with concatenated index structure
            and combined tensor type.
        Notes
        The resulting tensor has type
        ``(r1 + r2, s1 + s2)``, where ``(r1, s1)`` and ``(r2, s2)`` are the
        types of the two input tensors.
        """
        if not isinstance(other, LocalTensorField):
            raise TypeError("Tensor product is only defined between tensor fields.")

        if self.chart != other.chart:
            if self.rank<=other.rank:
                self_coord=self.transform_to(other.chart)
                other_coord=other
            else:
                self_coord=self
                other_coord=other.transform_to(self.chart)
        else:
            self_coord=self
            other_coord=other
        dim=self_coord.chart.dim
        left_rank=self_coord.rank
        right_rank=other_coord.rank
        product_rank=left_rank+right_rank
        left_get=self_coord.__getitem__
        right_get=other_coord.__getitem__
        data=[]

        for full_index in product(range(dim), repeat=(product_rank)):
            self_index=full_index[:left_rank]
            other_index=full_index[left_rank:]
            data.append(factor_terms(left_get(self_index)*right_get(other_index)))
        product_components=ImmutableDenseNDimArray(data, (dim,)*product_rank)
        return LocalTensorField(self_coord.chart, (self_coord.contravariant_order+other_coord.contravariant_order,
                                            self_coord.covariant_order+other_coord.covariant_order),
                                            product_components, self_coord.index_variance+other_coord.index_variance, None)

    def contraction(self, contravariant_index, covariant_index):
        """Contract one contravariant index with one covariant index.
        Parameters:
         - contravariant_index : int (Position of the contravariant index to contract)
         - covariant_index : int (Position of the covariant index to contract)
        Returns
        LocalTensorField: Contracted tensor field of type ``(r-1, s-1)``.
        """
        r=self.contravariant_order
        s=self.covariant_order
        dim=self.chart.dim
        total_rank=r+s

        if (contravariant_index<0 or contravariant_index>=total_rank) or (covariant_index<0 or covariant_index>=total_rank):
            raise ValueError("Invalid contraction indices (<0 or >r+s)")
        if(self.index_variance[contravariant_index]!=1):
            raise ValueError("contravariant_index must be a contravariant index")
        if(self.index_variance[covariant_index]!=-1):
            raise ValueError("covariant_index must be a covariant index")

        data=[]
        i=contravariant_index
        j=covariant_index
        if i>j: i,j=j,i
        tensor_get=self.__getitem__
        for reduced_index in product(range(dim),repeat=total_rank-2):
            suma=S.Zero
            for p in range(dim):
                full_index=(reduced_index[:i]+(p,)+reduced_index[i:j-1]+(p,)+reduced_index[j-1:])
                suma+=tensor_get(full_index)
            data.append(factor_terms(cancel(suma)))
        new_index_variance=tuple(v for i, v in enumerate(self.index_variance) if i not in (covariant_index, contravariant_index))
        return LocalTensorField(self.chart, (r-1,s-1), ImmutableDenseNDimArray(data, (dim,)*(total_rank-2)), new_index_variance)
    
    def __call__(self, *args):
        """Evaluate the tensor field on vectors and one-forms.
        Parameters:
         - *args : LocalTensorField
            Arguments to feed into the tensor. Each argument must itself be a
            local tensor field of rank one:
            - a ``(0, 1)`` tensor for contravariant slots,
            - a ``(1, 0)`` tensor for covariant slots.
        Returns:
         - LocalTensorField
            A rank-zero local tensor field containing the resulting scalar
            expression. If the tensor already has rank zero, it is returned
            unchanged.
        """
        r=self.contravariant_order
        s=self.covariant_order
        rank=r+s

        if(rank==0):
            return self
        if rank!=len(args):
            raise ValueError("The number of arguments does not match the rank of the tensor field")

        for i in range(rank):
            if not isinstance(args[i], LocalTensorField): raise ValueError("One of the arguments is not a Tensor Field"
            "(neither a Vector field nor a OneForm field)")
            if self.index_variance[i]==1 and args[i].tensor_type!=(0,1):
                 raise ValueError(f"Argument {i} must be a OneForm, since index {i} is contravariant.")
            if self.index_variance[i]==-1 and args[i].tensor_type!=(1,0):
                raise ValueError(f"Argument {i} must be a Vector, since index {i} is covariant.")
        tensor_get=self.__getitem__
        arg_components=[arg.components for arg in args]
        dim=self.chart.dim

        total=S.Zero
        for multi_index in product(range(self.chart.dim), repeat=self.rank):
            factors=[arg_components[a][multi_index[a]] for a in range(rank)]
            total+=Mul(*factors)*tensor_get(multi_index)
        return LocalTensorField(self.chart,(0,0),factor_terms(cancel(total)))
    
    def transform_to(self, new_chart):
        """Express the tensor field in another chart.
        Parameters:
         - new_chart : Chart (Target chart in which the tensor should be rewritten)
        Returns:
         - LocalTensorField
            Equivalent tensor field expressed in the target coordinates on the
            overlap of the source and target chart domains.
        Notes
        -----
        Contravariant indices are transformed with the Jacobian of the forward
        coordinate change, while covariant indices are transformed with the
        inverse Jacobian. Component expressions are also rewritten in the target
        coordinates. The resulting local tensor is defined only on the
        intersection of the two chart domains.
        """
        if new_chart==self.chart: return self
        if new_chart.dim!=self.chart.dim:
            raise ValueError("Dimensions of the old coordinate system and the new coordinate systems do not coincide")
        
        dim=self.chart.dim
        r=self.contravariant_order
        s=self.covariant_order
        rank=r+s
        J=self.chart.jacobian(new_chart)
        Jinv=new_chart.jacobian(self.chart)
        
        old_as_functions_of_new=new_chart.transform(self.chart)
        subs_rule=list(zip(self.chart.symbols, old_as_functions_of_new))
        subs_rule+=list(zip(self.chart.args[2], old_as_functions_of_new))
        tensor_get=self.__getitem__
        index_variance=self.index_variance
        
        contra_positions=[a for a in range(rank) if self.index_variance[a]==1]
        cov_positions=[a for a in range(rank) if self.index_variance[a]==-1]
        all_indices=tuple(product(range(dim), repeat=(r+s)))

        if self.equivalences is None: target_indices=all_indices
        else: target_indices=sorted(index for index in all_indices if index not in self.equivalences)
        
        J_sub={(i,j):J[i,j].subs(subs_rule) for i in range(dim) for j in range(dim)}
        tensor_sub={idx: _sympify(tensor_get(idx)).subs(subs_rule) for idx in all_indices}
        data=[]
        for multi_index in target_indices:
            suma=S.Zero
            for sum_index in all_indices:
                factor=tensor_sub[sum_index]
                for a in contra_positions: factor*=J_sub[multi_index[a], sum_index[a]]
                for a in cov_positions: factor*=Jinv[sum_index[a], multi_index[a]]
                suma+=factor
            data.append(factor_terms(cancel(suma)))
        
        W=self.chart.open_set.intersection(new_chart.open_set)
        target_chart=new_chart if W==new_chart.open_set else new_chart.restrict(W)
        if target_chart!=new_chart:
            target_subs=dict(zip(new_chart.symbols, target_chart.symbols))
            target_subs.update(dict(zip(new_chart.args[2], target_chart.symbols)))
            data=[_sympify(expr).xreplace(target_subs) for expr in data]
        
        if self.equivalences is None: components=ImmutableDenseNDimArray(data,(dim,)*(r+s))
        else: components=tuple(data)
        return LocalTensorField(target_chart,(r,s),components, self.index_variance, self.equivalences)
    
    def to_matrix(self):
        """Return the matrix representation of a rank-two tensor.
        Returns: ImmutableDenseMatrix
            Matrix whose entries are the tensor components in the current chart.
        Notes:
        This method is intended for tensor types ``(0, 2)``, ``(1, 1)``, and
        ``(2, 0)``.
        """
        if self.rank!=2:
            raise ValueError("to_matrix only return the matrix M representing a (0,2), (1,1) or (2,0) tensor field")
        dim=self.chart.dim
        return ImmutableDenseMatrix([[self[(i,j)] for j in range(dim)] for i in range(dim)])
    
    def raise_lower_indices(self, indices, g_cov, g_contra):
        """Raise or lower the selected indices using local metric tensors."""
        if not indices: return self
        r=self.contravariant_order
        s=self.covariant_order
        rank=r+s
        dim=self.chart.dim
        if any(i<0 or i>=rank for i in indices):
            raise ValueError("Some index is out of range.")
        if len(set(indices))!=len(indices):
            raise ValueError("Repeated indices are not allowed.")

        indices=sorted(indices)
        new_index_variance=list(self.index_variance)
        for i in indices: new_index_variance[i]*=-1
        new_index_variance=tuple(new_index_variance)
        data=[]
        for out_index in product(range(dim), repeat=(rank)):
            suma=S.Zero
            for dummy_index in product(range(dim), repeat=len(indices)):
                old_index=list(out_index)
                factor=S.One
                for t, pos in enumerate(indices):
                    if self.index_variance[pos]==-1:
                        factor*=g_contra[out_index[pos], dummy_index[t]]
                    else:
                        factor*=g_cov[out_index[pos], dummy_index[t]]
                    old_index[pos]=dummy_index[t]
                suma+=self[tuple(old_index)]*factor
            data.append(suma)
        r_new=sum(1 for v in new_index_variance if v==1)
        s_new=rank-r_new
        return LocalTensorField(self.chart, (r_new,s_new), ImmutableDenseNDimArray(data, (dim,)*rank), new_index_variance)

    def permute_indices(self,order):
        rank=self.rank
        if tuple(sorted(order))!=tuple(range(rank)):
            raise ValueError("order must be a permutation of the indices")
        dim=self.chart.dim
        data=[]
        for new_index in product(range(dim),repeat=rank):
            old_index=tuple(new_index[order[i]] for i in range(rank))
            data.append(self[old_index])
        new_index_variance=tuple(self.index_variance[order[i]] for i in range(rank))
        return LocalTensorField(self.chart, (self.contravariant_order,self.covariant_order), ImmutableDenseNDimArray(data, (dim,)*rank),new_index_variance,None)

from tensorium.fields.tensor_field import TensorField

def _is_tensor_field_component(obj):
    """Test whether an object can be used as a tensor component.
    Parameters:
     - obj : object
        Candidate object.
    Returns:
     - bool
        ``True`` if the object behaves like a ``TensorField``.
    """
    return isinstance(obj, TensorField) or (
        hasattr(obj, "local_representations")
        and hasattr(obj, "local_representation")
        and hasattr(obj, "tensor_type")
        and hasattr(obj, "index_variance")
    )

class ValuedTensorField:
    """Tensor field valued in fixed finite-dimensional internal spaces.

    This class models tensor fields whose components carry extra internal
    vector-space indices in a chosen trivialization. Equivalently, it represents
    sections of a trivialized bundle of the form
    ``T^r_s M tensor E_1 tensor ... tensor E_k``. It does not implement
    general non-trivial vector bundles or changes of internal frame.
    """
    def __init__(self,components,internal_shape,internal_variance):
        internal_shape=tuple(internal_shape)
        internal_variance=tuple(internal_variance)
        if len(internal_shape)!=len(internal_variance): raise ValueError("internal_shape and internal_variance must have the same length")
        if any(dim<=0 for dim in internal_shape): raise ValueError("internal_shape dimensions must be positive")
        actual_shape=nested_shape(components)
        if actual_shape!=internal_shape: raise ValueError(f"components have internal shape {actual_shape}, "f"but internal_shape={internal_shape} was requested")                                                                       
        first=first_tensor_component(components)
        validate_components(components,first)
        self._components=freeze_nested(components)
        self._internal_shape=internal_shape
        self._internal_variance=internal_variance
        self._first_component=first
    @property
    def components(self): return self._components
    @property
    def internal_shape(self): return self._internal_shape
    @property
    def internal_variance(self): return self._internal_variance
    @property
    def internal_rank(self): return len(self.internal_shape)
    @property
    def base_manifold(self): return self._first_component.base_manifold
    @property
    def manifold(self): return self._first_component.manifold
    @property
    def tensor_type(self): return self._first_component.tensor_type
    @property
    def index_variance(self): return self._first_component.index_variance
    @property
    def rank(self): return self._first_component.rank
    @property
    def signature(self):
        """Return the field signature associated with this field with internal values.
        Returns:
         - FieldSignature
            Signature containing geometric and internal index data.
        """
        from tensorium.operators.signature import FieldSignature
        internal_variance=self.internal_variance
        if isinstance(internal_variance,int): internal_variance=(internal_variance,)
        return FieldSignature(tensor_type=self.tensor_type,index_variance=self.index_variance,
                              internal_shape=self.internal_shape,internal_variance=internal_variance,manifold=self.base_manifold)

    def __len__(self): return self.internal_shape[0]
    def __iter__(self): return iter(self.components)
    def __repr__(self):
        return (f"{self.__class__.__name__}(internal_shape={self.internal_shape}, "
                f"internal_variance={self.internal_variance}, tensor_type={self.tensor_type})")
    def __getitem__(self,indices):
        """Return the component selected by internal indices.
        Parameters:
         - indices : int or tuple
            Internal index or tuple of internal indices.
        Returns:
         - TensorField
            Tensor component at the selected internal position.
        """
        if not isinstance(indices,tuple): indices=(indices,)
        component=self.components
        for index in indices: component=component[index]
        return component

    def __add__(self,other):
        """Add valued tensor fields component by component.
        Parameters:
         - other : ValuedTensorField
            Field with the same internal shape, internal variance and geometric
            tensor type.
        Returns:
         - ValuedTensorField
            Componentwise sum.
        """
        check=self._check_same_indexed_structure(other,operation="add")
        if check is NotImplemented: return NotImplemented
        return self._new_like(nested_zip_map(lambda f1,f2: f1+f2,self.components,other.components))

    def __neg__(self):
        """Return the additive inverse component by component.
        Returns:
         - ValuedTensorField
            Field with all components multiplied by ``-1``.
        """
        return self._new_like(nested_map(lambda f: -f,self.components))
    def __sub__(self,other):
        """Subtract valued tensor fields component by component.
        Parameters:
         - other : ValuedTensorField
            Field to subtract.
        Returns:
         - ValuedTensorField
            Componentwise difference.
        """
        return self+(-other)
    def __mul__(self,scalar):
        """Multiply every component by a scalar-like object.
        Parameters:
         - scalar : object
            Scalar-like factor applied on the right.
        Returns:
         - ValuedTensorField
            Field with scaled components.
        """
        if isinstance(scalar,ValuedTensorField): return NotImplemented
        return self._new_like(nested_map(lambda f: f*scalar,self.components))
    def __rmul__(self,scalar):
        """Multiply every component by a scalar-like object on the left.
        Parameters:
         - scalar : object
            Scalar-like factor applied on the left.
        Returns:
         - ValuedTensorField
            Field with scaled components.
        """
        if isinstance(scalar,ValuedTensorField): return NotImplemented
        return self._new_like(nested_map(lambda f: scalar*f,self.components))

    def local_representation(self,chart):
        """Return all components represented in a chart.
        Parameters:
         - chart : Chart
            Chart in which each tensor component should be expressed.
        Returns:
         - tuple
            Nested tuple of local tensor components.
        """
        return nested_map(lambda f: f.local_representation(chart),self.components)

    def tensor_product(self,other):
        """Return the external tensor product.
        Parameters:
         - other : TensorField or ValuedTensorField
            Second factor in the tensor product.
        Returns:
         - ValuedTensorField
            Tensor product preserving all internal indices. If ``other`` is an
            ``ValuedTensorField``, the internal indices of ``other`` are
            appended after those of ``self``.
        """
        if _is_tensor_field_component(other):
            if self.base_manifold!=other.base_manifold: raise ValueError("cannot take tensor product of indexed tensor field and tensor field defined on different base manifolds")
            return self._new_like(nested_map(lambda f: f.tensor_product(other),self.components))
        if isinstance(other,ValuedTensorField):
            if self.base_manifold!=other.base_manifold: raise ValueError("cannot take tensor product of indexed tensor fields defined on different base manifolds")
            def outer(left,right):
                if _is_tensor_field_component(left) and _is_tensor_field_component(right): return left.tensor_product(right)
                if _is_tensor_field_component(left): return tuple(outer(left,component) for component in right)
                if _is_tensor_field_component(right): return tuple(outer(component,right) for component in left)
                return tuple(outer(l,right) for l in left)
            return self._new_like(outer(self.components,other.components),self.internal_shape+other.internal_shape,
                self._internal_variance+other._internal_variance)
        return NotImplemented
    
    def _check_same_indexed_structure(self,other,operation="operate on"):
        """Validate compatibility for componentwise operations.
        Parameters:
         - other : object
            Candidate ``ValuedTensorField``.
         - operation : str, optional
            Description used in error messages.
        Returns:
         - None or NotImplemented
            ``None`` when the structures are compatible, ``NotImplemented`` if
            ``other`` is not an ``ValuedTensorField``.
        """
        if not isinstance(other,ValuedTensorField): return NotImplemented
        if self.internal_shape!=other.internal_shape: raise ValueError(f"cannot {operation} indexed tensor fields with different internal shapes")
        if self._internal_variance != other._internal_variance:
            raise ValueError(f"cannot {operation} indexed tensor fields with different internal variances")
        if self.base_manifold != other.base_manifold:
            raise ValueError(f"cannot {operation} indexed tensor fields defined on different base manifolds")
        if self.tensor_type != other.tensor_type:
            raise ValueError(f"cannot {operation} indexed tensor fields with different tensor types")
        if self.index_variance != other.index_variance:
            raise ValueError(f"cannot {operation} indexed tensor fields with different index variances")
        return None

    def _new_like(self, components, internal_shape=None, internal_variance=None):
        """Build a new object of the same class when possible.
        Parameters:
         - components : nested tuple
            New component array.
         - internal_shape : tuple, optional
            Internal shape of the new object.
         - internal_variance : tuple, optional
            Internal variance pattern of the new object.
        Returns:
         - ValuedTensorField
            New valued tensor field.
        """
        return self.__class__(components,self.internal_shape if internal_shape is None else internal_shape,
            self.internal_variance if internal_variance is None else internal_variance,)

    def internal_contraction(self,internal_index1,internal_index2):
        """Contract two opposite-variance internal indices.
        Parameters:
         - internal_index1 : int
            First internal index position.
         - internal_index2 : int
            Second internal index position.
        Returns:
         - ValuedTensorField or TensorField
            Result of contracting the selected internal pair. If no internal
            indices remain, the resulting tensor field is returned directly.
        """
        i,j=internal_index1,internal_index2
        rank=self.internal_rank
        if not isinstance(i,int) or not isinstance(j,int): raise TypeError("internal contraction indices must be integers")
        if i<0 or i>=rank or j<0 or j>=rank: raise IndexError("internal_index out of range")
        if i==j: raise ValueError("internal_index1 and internal_index2 must be different")
        if self.internal_shape[i]!=self.internal_shape[j]: raise ValueError("internal indices must have the same dimension to be contracted")
        if self._internal_variance[i]==self._internal_variance[j]: raise ValueError("internal indices must have opposite variance to be contracted")
        keep_positions=[k for k in range(rank) if k!=i and k!=j]
        new_internal_shape=tuple(self.internal_shape[k] for k in keep_positions)
        new_internal_variance=tuple(self._internal_variance[k] for k in keep_positions)
        def build_component(kept_indices):
            total=None
            for contracted_index in range(self.internal_shape[i]):
                full_index=[None]*rank
                for pos,value in zip(keep_positions,kept_indices): full_index[pos]=value
                full_index[i]=contracted_index
                full_index[j]=contracted_index
                component=self[tuple(full_index)]
                if total is None: total=component
                else: total+=component
            return total
        new_components=_build_nested_components(new_internal_shape,build_component)
        if new_internal_shape==(): return new_components
        return ValuedTensorField(new_components,new_internal_shape,new_internal_variance)

    def internal_action(self,other,pairs=None):
        """Apply this internal tensor to another one by internal contractions.
        Parameters:
         - other : ValuedTensorField
            Field on which ``self`` acts.
         - pairs : iterable, optional
            Pairs ``(i,j)`` indicating that internal index ``i`` of ``self`` is
            contracted with internal index ``j`` of ``other``. If omitted, the
            method tries to infer a unique compatible pair.
        Returns:
         - ValuedTensorField or TensorField
            Tensor product followed by the requested internal contractions.
        """
        if not isinstance(other,ValuedTensorField): return NotImplemented
        if self.base_manifold!=other.base_manifold: raise ValueError("cannot apply valued tensor fields defined on different base manifolds")
        if pairs is None: pairs=_default_internal_action_pairs(self,other)
        product=self.tensor_product(other)
        shifted_pairs=[(i,j+len(self.internal_shape)) for i,j in pairs]
        while shifted_pairs:
            i,j=max(shifted_pairs,key=lambda pair: max(pair))
            product=product.internal_contraction(i,j)
            shifted_pairs=[
                (_shift_index_after_internal_contraction(a,i,j),_shift_index_after_internal_contraction(b,i,j))
                for a,b in shifted_pairs
                if (a,b)!=(i,j)
            ]
        return product

    def raise_lower_indices(self,indices):
        """Raise or lower geometric indices component by component.
        Parameters:
         - indices : iterable
            Geometric index positions to raise or lower.
        Returns:
         - ValuedTensorField
            Field whose tensor components have the requested indices changed.
        """
        return self._new_like(nested_map(lambda component: component.raise_lower_indices(indices),self.components,))

    def permute_geometric_indices(self,other):
        """Permute geometric indices component by component.
        Parameters:
         - other : tuple
            Permutation passed to each tensor component.
        Returns:
         - ValuedTensorField
            Field with permuted geometric indices.
        """
        return self._new_like(nested_map(lambda f: f.permute_indices(other),self.components))
    def permute_internal_indices(self,order):
        """Permute internal indices according to ``order``.
        Parameters:
         - order : tuple
            Permutation of internal index positions. The entry
            ``order[new_pos]`` gives the old position moved to ``new_pos``.
        Returns:
         - ValuedTensorField
            Field with internal indices permuted and geometric indices left
            unchanged.
        """
        order=tuple(order)
        if len(order)!=self.internal_rank: raise ValueError("order must have one entry per internal index")
        if sorted(order)!=list(range(self.internal_rank)): raise ValueError("order must be a permutation of internal indices")
        new_internal_shape=tuple(self.internal_shape[i] for i in order)
        new_internal_variance=tuple(self.internal_variance[i] for i in order)
        def build_component(new_indices):
            old_indices=[None]*self.internal_rank
            for new_pos,old_pos in enumerate(order): old_indices[old_pos]=new_indices[new_pos]
            return self[tuple(old_indices)]
        new_components=_build_nested_components(new_internal_shape,build_component)
        return self._new_like(new_components,internal_shape=new_internal_shape,internal_variance=new_internal_variance)
    

    def contraction(self,contravariant_index,covariant_index):
        """Contract geometric indices component by component.
        Parameters:
         - contravariant_index : int
            Position of the contravariant geometric index.
         - covariant_index : int
            Position of the covariant geometric index.
        Returns:
         - ValuedTensorField
            Field obtained by contracting each tensor component.
        """
        return self._new_like(nested_map(lambda component: component.contraction(contravariant_index, covariant_index),self.components,))



def nested_shape(components):
    """Return the shape of a nested component array.
    Parameters:
     - components : nested list, tuple, or TensorField
        Nested component structure.
    Returns:
     - tuple
        Shape of the nested structure, excluding the tensor-field leaf.
    """
    if _is_tensor_field_component(components): return ()
    if not isinstance(components,(list,tuple)): raise TypeError("components must be a nested list or tuple of TensorFields")
    if not components: raise ValueError("components cannot be empty")
    first_shape=nested_shape(components[0])
    for c in components[1:]:
        if nested_shape(c)!=first_shape: raise ValueError("all components must have the same nested shape")
    return (len(components),)+first_shape
    
def first_tensor_component(components):
    """Return the first tensor-field-like component in a nested array.
    Parameters:
     - components : nested list, tuple, or TensorField
        Nested component structure.
    Returns:
     - TensorField
        First tensor-field-like leaf.
    """
    if _is_tensor_field_component(components): return components
    if not isinstance(components,(list,tuple)): raise TypeError("components must be a nested list or tuple of TensorFields")
    if not components: raise ValueError("components cannot be empty")
    return first_tensor_component(components[0])
    
def iter_tensor_components(components):
    """Iterate over tensor-field-like leaves in a nested component array.
    Parameters:
     - components : nested list, tuple, or TensorField
        Nested component structure.
    Yields:
     - TensorField
        Tensor-field-like leaves in the nested structure.
    """
    if _is_tensor_field_component(components): yield components
    elif isinstance(components,(list,tuple)):
        for c in components:
            yield from iter_tensor_components(c)
    else: raise TypeError("components must be a nested list or tuple of TensorFields")

def validate_components(components, first=None):
    """Validate that all nested components are compatible tensor fields.
    Parameters:
     - components : nested list, tuple, or TensorField
        Nested component structure to validate.
     - first : TensorField, optional
        Reference component. If omitted, the first component is used.
    Returns:
     - None
    """
    if first is None: first=first_tensor_component(components)
    if not _is_tensor_field_component(first): raise TypeError("components must be a nested list or tuple of TensorFields")
    for c in iter_tensor_components(components):
        if not _is_tensor_field_component(c): raise TypeError("all components must be TensorFields")
        if c.tensor_type!=first.tensor_type: raise ValueError("all components must have the same tensor type")
        if c.index_variance!=first.index_variance: raise ValueError("all components must have the same index variance")
        if c.base_manifold!=first.base_manifold: raise ValueError("all components must be defined on the same base manifold")

def freeze_nested(components):
    """Convert nested component lists into tuples.
    Parameters:
     - components : nested list, tuple, or TensorField
        Nested component structure.
    Returns:
     - tuple or TensorField
        Immutable nested tuple structure with the same leaves.
    """
    if _is_tensor_field_component(components): return components
    return tuple(freeze_nested(c) for c in components)

def nested_map(func, components):
    """Map a function over tensor-field leaves in a nested array.
    Parameters:
     - func : callable
        Function applied to each tensor-field-like leaf.
     - components : nested tuple or TensorField
        Nested component structure.
    Returns:
     - tuple or object
        Nested result with the same shape.
    """
    if _is_tensor_field_component(components): return func(components)
    return tuple(nested_map(func,component) for component in components)

def nested_zip_map(func,left,right):
    """Zip two nested component arrays and map a function over their leaves.
    Parameters:
     - func : callable
        Function applied to pairs of tensor-field-like leaves.
     - left : nested tuple or TensorField
        First nested component structure.
     - right : nested tuple or TensorField
        Second nested component structure.
    Returns:
     - tuple or object
        Nested result with the same shape.
    """
    if _is_tensor_field_component(left) and _is_tensor_field_component(right): return func(left,right)
    if _is_tensor_field_component(left) or _is_tensor_field_component(right): raise ValueError("nested structures do not have the same shape")
    return tuple(nested_zip_map(func,l,r) for l,r in zip(left,right))

def _build_nested_components(shape,build_component):
    """Build a nested tuple of components with a prescribed shape.
    Parameters:
     - shape : tuple
        Desired nested shape.
     - build_component : callable
        Function receiving an index tuple and returning the corresponding leaf.
    Returns:
     - tuple or object
        Nested component structure.
    """
    if shape==(): return build_component(())
    return tuple(_build_nested_components(shape[1:],lambda rest_indices, index=index: build_component((index,)+rest_indices),) for index in range(shape[0]))

def _default_internal_action_pairs(left,right):
    """Infer the unique compatible pair of internal indices to contract.
    Parameters:
     - left : ValuedTensorField
        Left valued tensor field.
     - right : ValuedTensorField
        Right valued tensor field.
    Returns:
     - list
        Single pair ``(i,j)`` contracting an index of ``left`` with one of
        ``right``.
    """
    candidates=[]
    for i,variance_left in enumerate(left._internal_variance):
        for j,variance_right in enumerate(right._internal_variance):
            if left.internal_shape[i]==right.internal_shape[j] and variance_left!=variance_right:
                candidates.append((i,j))
    if len(candidates)!=1:
        raise ValueError("could not infer a unique internal contraction pair; pass pairs explicitly")
    return candidates

def _shift_index_after_internal_contraction(index,i,j):
    """Update an internal index position after contracting two positions.
    Parameters:
     - index : int
        Original internal index position.
     - i : int
        First contracted position.
     - j : int
        Second contracted position.
    Returns:
     - int
        New index position after removing ``i`` and ``j``.
    """
    first,second=sorted((i,j))
    if index==first or index==second: raise ValueError("cannot shift a contracted index")
    return index-(index>first)-(index>second)

class TensorMultiplet(ValuedTensorField):
    """Rank-one with internal values tensor field in a fixed internal basis."""
    def __init__(self,components,internal_variance=1):
        """Create a rank-one valued tensor field.
        Parameters:
         - components : list or tuple
            Tensor-field components of the multiplet.
         - internal_variance : int, optional
            Variance of the internal index. Use ``1`` for contravariant and
            ``-1`` for covariant.
        Returns:
         - TensorMultiplet
            Internal rank-one tensor field.
        """
        super().__init__(components,(len(components),),(internal_variance,))
    @property
    def dim_internal(self): return self.internal_shape[0]
    @property
    def internal_variance(self): return super().internal_variance[0]

    def _new_like(self, components, internal_shape=None, internal_variance=None):
        """Build a new multiplet when the internal rank remains one.
        Parameters:
         - components : nested tuple
            New component array.
         - internal_shape : tuple, optional
            Internal shape of the result.
         - internal_variance : tuple, optional
            Internal variance pattern of the result.
        Returns:
         - TensorMultiplet or ValuedTensorField
            ``TensorMultiplet`` if the result still has one internal index,
            otherwise a general ``ValuedTensorField``.
        """
        if internal_shape is None: internal_shape=self.internal_shape
        if internal_variance is None: internal_variance=(self.internal_variance,)
        if internal_shape==self.internal_shape and len(internal_variance)==1:
            return TensorMultiplet(components,internal_variance[0])
        return ValuedTensorField(components,internal_shape,internal_variance)

class FieldSignature:
    """Lightweight description of the shape of a tensor-like field.

    A ``FieldSignature`` is not a tensor field. It only records the geometric
    tensor type, the order of geometric indices when known, and the finite
    internal slots carried by a field with internal values. Generic signatures
    use symbolic labels such as ``("r","s")`` and are meant for display and
    operator bookkeeping rather than strict type checking.
    """
    def __init__(self,tensor_type=None,index_variance=None,internal_shape=(),
                 internal_variance=(),manifold=None,generic=False):
        """Create a field signature.
        Parameters:
         - tensor_type : tuple, optional
            Pair ``(r,s)`` or symbolic pair such as ``("r","s")``.
         - index_variance : tuple, optional
            Ordered geometric variance pattern. Use ``1`` for contravariant
            and ``-1`` for covariant indices.
         - internal_shape : tuple, optional
            Dimensions or symbolic labels of the internal slots.
         - internal_variance : tuple, optional
            Variance pattern of the internal slots.
         - manifold : Manifold, optional
            Manifold on which the field is defined.
         - generic : bool, optional
            Whether this is a symbolic generic signature.
        Returns:
         - FieldSignature
            Shape descriptor for a tensor-like field.
        """
        self.tensor_type=tensor_type
        self.index_variance=None if index_variance is None else tuple(index_variance)
        self.internal_shape=None if internal_shape is None else tuple(internal_shape)
        self.internal_variance=None if internal_variance is None else tuple(internal_variance)
        self.manifold=manifold
        self.generic=generic
        if self.index_variance is not None and self.tensor_type is not None:
            if len(self.index_variance)!=sum(self.tensor_type): raise ValueError("index_variance length must match tensor_type")
        if self.internal_shape is not None and self.internal_variance is not None:
            if len(self.internal_shape)!=len(self.internal_variance): raise ValueError("internal_shape and internal_variance must have the same length")
    @classmethod
    def generic_tensor(cls,manifold=None):
        """Return the generic signature ``T^r_s(M)``."""
        return cls(tensor_type=("r","s"),manifold=manifold,generic=True)
    @classmethod
    def generic_internal_tensor(cls,manifold=None):
        """Return the generic with internal values signature.

        The symbolic internal labels ``("a","b")`` represent an arbitrary
        number of upper and lower internal slots. This is only a display-level
        convention; concrete fields still store explicit internal shapes.
        """
        return cls(tensor_type=("r","s"),internal_shape=("a","b"),internal_variance=(1,-1),manifold=manifold,generic=True)
    def with_extra_geometric_indices(self,variance):
        """Return the signature obtained by appending free geometric indices.
        Parameters:
         - variance : tuple
            Variance pattern of the added geometric indices.
        Returns:
         - FieldSignature
            Signature with updated tensor type and ordered index variance.
        """
        variance=tuple(variance)
        if self.tensor_type is None: tensor_type=None
        else:
            r,s=self.tensor_type
            tensor_type = (_add_symbolic(r, sum(1 for v in variance if v == 1)),_add_symbolic(s, sum(1 for v in variance if v == -1)),)
        if self.index_variance is None: index_variance=None
        else: index_variance=self.index_variance+variance
        return FieldSignature(tensor_type=tensor_type,index_variance=index_variance,internal_shape=self.internal_shape,
                                  internal_variance=self.internal_variance,manifold=self.manifold,generic=self.generic)

    def without_geometric_indices(self,positions):
        positions=tuple(sorted(set(positions)))
        if self.index_variance is None:
            index_variance=None
            removed_variance=None
        else:
            if any(pos<0 or pos>=len(self.index_variance) for pos in positions):  raise IndexError("geometric signature index out of range")
            removed_variance=tuple(self.index_variance[pos] for pos in positions)
            index_variance=tuple(v for pos,v in enumerate(self.index_variance) if pos not in positions)
        if self.tensor_type is None: tensor_type=None
        elif removed_variance is None: tensor_type=self.with_tensor_type_shift(-1,-1).tensor_type
        else:
            r,s=self.tensor_type
            tensor_type=(_add_symbolic(r,-sum(1 for v in removed_variance if v==1)),_add_symbolic(s,-sum(1 for v in removed_variance if v==-1)))
        return FieldSignature(tensor_type=tensor_type,index_variance=index_variance,internal_shape=self.internal_shape,
                          internal_variance=self.internal_variance,manifold=self.manifold,generic=self.generic)
    
    def __repr__(self): return (f"FieldSignature(tensor_type={self.tensor_type!r}, "
                                f"index_variance={self.index_variance!r}, "
                                f"internal_shape={self.internal_shape!r}, "
                                f"internal_variance={self.internal_variance!r}, "
                                f"generic={self.generic!r})")
    def with_tensor_type_shift(self,dr=0,ds=0):
        """Return a copy with shifted geometric tensor type.
        Parameters:
         - dr : int
            Shift applied to the contravariant tensor rank.
         - ds : int
            Shift applied to the covariant tensor rank.

        Returns:
         - FieldSignature
            New signature with tensor type ``(r+dr, s+ds)``.
        """
        if self.tensor_type is None: tensor_type=None
        else:
            r,s=self.tensor_type
            tensor_type=(_add_symbolic(r,dr),_add_symbolic(s,ds))
        return FieldSignature(tensor_type=tensor_type,index_variance=self.index_variance,internal_shape=self.internal_shape,
                              internal_variance=self.internal_variance,manifold=self.manifold,generic=self.generic)
    def compatible_with(self,other):
        """Return whether this signature can describe the same kind of field as another one.
        Parameters:
         - other : FieldSignature
            Signature to compare with.
        Returns:
         - bool
            ``True`` when all known geometric and internal data are compatible.
        """
        if not isinstance(other,FieldSignature): return False
        if self.manifold is not None and other.manifold is not None and self.manifold is not other.manifold: return False
        if not _compatible_tensor_type(self.tensor_type,other.tensor_type): return False
        if not _compatible_tuple(self.index_variance,other.index_variance): return False
        if not _compatible_internal_data(self,other): return False
        return True

class OperatorSignature:
    """Input/output signature of a tensor operator.

    The output is stored as a rule instead of a fixed signature because many
    operators depend on the type of their inputs: tensor product, contraction,
    covariant derivatives and internal actions all transform signatures.
    """
    def __init__(self,inputs,output_rule,free_index_variance=()):
        """Create an operator signature.
        Parameters:
         - inputs : iterable
            Expected input ``FieldSignature`` objects. They may be generic.
         - output_rule : callable
            Function receiving input signatures and returning the output
            ``FieldSignature``.
         - free_index_variance : tuple, optional
            Free geometric indices carried by the operator itself.
        Returns:
         - OperatorSignature
            Lightweight descriptor for a tensor operator.
        """
        self.inputs=tuple(inputs)
        self.output_rule=output_rule
        self.free_index_variance=tuple(free_index_variance)

    @property
    def arity(self): return len(self.inputs)

    def output_for(self,*input_signatures):
        """Evaluate the output rule on concrete or generic input signatures."""
        self.validate_inputs(*input_signatures)
        return self.output_rule(*input_signatures)

    def __repr__(self): return (f"OperatorSignature(arity={self.arity}, "
                                f"free_index_variance={self.free_index_variance!r})")

    def accepts(self,*input_signatures):
        """Return whether the provided field signatures are valid inputs.
        Parameters:
         - input_signatures : tuple
            Candidate input signatures.
        Returns:
         - bool
            ``True`` if the arity is correct and each input is compatible with
            the corresponding expected signature.
        """
        if len(input_signatures)!=self.arity: return False
        return all(expected.compatible_with(actual) for expected,actual in zip(self.inputs,input_signatures))

    def validate_inputs(self,*input_signatures):
        """Raise an error if the provided field signatures are not valid inputs.
        Parameters:
         - input_signatures : tuple
            Candidate input signatures.
        Returns:
         - None
        """
        if len(input_signatures)!=self.arity: raise ValueError(f"operator expects {self.arity} input(s), got {len(input_signatures)}")
        for pos,(expected,actual) in enumerate(zip(self.inputs,input_signatures)):
            if not expected.compatible_with(actual): raise ValueError(f"input {pos} has incompatible signature: expected {expected}, got {actual}")

def _add_symbolic(value, shift):
    """Add an integer shift to an integer or symbolic tensor-rank label."""
    if shift==0: return value
    if isinstance(value,int): return value+shift
    if isinstance(value,str):
        for sign in ("+","-"):
            if sign in value[1:]:
                base,offset=value.rsplit(sign,1)
                if offset.isdigit():
                    total=(int(offset) if sign=="+" else -int(offset))+shift
                    return _add_symbolic(base,total)
    if shift>0: return f"{value}+{shift}" if shift>1 else f"{value}+1"
    return f"{value}{shift}" if shift<-1 else f"{value}-1"

def _compatible_entry(left,right,allow_symbolic=False):
    """Return whether two signature entries are compatible."""
    if left is None or right is None: return True
    if allow_symbolic and (isinstance(left,str) or isinstance(right,str)): return True
    return left==right

def _compatible_tuple(left,right,allow_symbolic=False):
    """Return whether two optional tuples are componentwise compatible."""
    if left is None or right is None: return True
    left=tuple(left)
    right=tuple(right)
    if len(left)!=len(right): return False
    return all(_compatible_entry(a,b,allow_symbolic=allow_symbolic) for a,b in zip(left,right))

def _compatible_tensor_type(left,right):
    """Return whether two tensor types are compatible."""
    if left is None or right is None: return True
    if len(left)!=2 or len(right)!=2: return False
    return all(_compatible_entry(a,b,allow_symbolic=True) for a,b in zip(left,right))

def _is_generic_internal_wildcard(signature):
    """Return whether a signature denotes an arbitrary field with internal values."""
    return (getattr(signature,"generic",False)
            and tuple(getattr(signature,"internal_shape",()))==("a","b")
            and tuple(getattr(signature,"internal_variance",()))==(1,-1))

def _compatible_internal_data(left,right):
    """Return whether the internal parts of two field signatures are compatible."""
    if left.internal_shape is None or right.internal_shape is None: return True
    if left.internal_variance is None or right.internal_variance is None: return True
    left_wildcard=_is_generic_internal_wildcard(left)
    right_wildcard=_is_generic_internal_wildcard(right)
    if left_wildcard or right_wildcard:
        left_has_internal=bool(left.internal_shape) or left_wildcard
        right_has_internal=bool(right.internal_shape) or right_wildcard
        return left_has_internal and right_has_internal
    if not _compatible_tuple(left.internal_shape,right.internal_shape,allow_symbolic=True): return False
    if not _compatible_tuple(left.internal_variance,right.internal_variance): return False
    return True
        

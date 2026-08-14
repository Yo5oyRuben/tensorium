from tensorium.operators.signature import FieldSignature, OperatorSignature

class TensorOperator:
    """Tensor-valued operator with free geometric indices.
    The operator stores an index variance pattern and an action on tensor
    fields. Composite operators are represented by new ``TensorOperator``
    instances rather than by dedicated subclasses.
    """
    def __init__(self,index_variance=(),action=None,name=None, arity=1, signature=None,operator_kind="generic", metadata=None):
        """Create a tensor operator.
        Parameters:
         - index_variance : tuple, optional
            Variance pattern of the free geometric indices carried by the
            operator. Use ``1`` for contravariant indices and ``-1`` for
            covariant ones.
         - action : callable, optional
            Function implementing the action of the operator.
         - name : str, optional
            Symbolic name used for representation and display.
         - arity : int, optional
            Number of arguments accepted by the operator.
         - operator_kind : str, optional
            Lightweight semantic label used for display.
         - metadata : dict, optional
            Extra non-computational information used by display utilities.
        Returns:
         - TensorOperator (Tensor-valued operator with the requested action)
        """
        self._index_variance=tuple(index_variance)
        self._action=action
        self.name=name
        self._arity=arity
        self._operator_kind=operator_kind
        self._metadata={} if metadata is None else dict(metadata)
        self._signature=signature
    @property
    def index_variance(self): return self._index_variance
    @property
    def rank(self): return len(self.index_variance)
    @property
    def tensor_type(self):
        return(sum(1 for v in self.index_variance if v==1),sum(1 for v in self.index_variance if v==-1))
    @property
    def arity(self): return self._arity
    @property
    def is_scalar_operator(self): return self.rank==0
    @property
    def operator_kind(self): return self._operator_kind
    @property
    def metadata(self): return dict(self._metadata)
    @property
    def signature(self): return self._signature
    @staticmethod
    def _generic_unary_signature(free_index_variance=()):
        field=FieldSignature.generic_tensor()
        return OperatorSignature(inputs=(field,),output_rule=lambda F: F.with_extra_geometric_indices(free_index_variance),free_index_variance=free_index_variance,)
    @staticmethod
    def _generic_internal_unary_signature(free_index_variance=()):
        field=FieldSignature.generic_internal_tensor()
        return OperatorSignature(inputs=(field,),output_rule=lambda F: F.with_extra_geometric_indices(free_index_variance),free_index_variance=free_index_variance)

    def __repr__(self):
        return f"TensorOperator(name={self.name!r}, index_variance={self.index_variance})"
    def __call__(self,*args):
        """Apply the operator.
        Parameters:
         - args : tuple
            Arguments passed to the stored action. For most operators this is a
            single tensor field.
        Returns:
         - TensorField or ValuedTensorField
            Result of applying the operator.
        """
        if self._action is None: raise NotImplementedError
        if self.signature is not None:
            arg_signatures=[]
            for pos,arg in enumerate(args):
                if not hasattr(arg,"signature"): raise ValueError(f"argument {pos} has no field signature")
                arg_signatures.append(arg.signature)
            self.signature.validate_inputs(*arg_signatures)
        return self._action(*args)
    
    def compose(self,other):
        """Compose two unary tensor operators.
        Parameters:
         - other : TensorOperator
            Operator applied first.
        Returns:
         - TensorOperator
            Composite operator ``self∘other``. The free indices of ``other``
            are followed by the free indices of ``self``.
        """
        if not isinstance(other,TensorOperator): raise ValueError("can only compose tensor operators")
        if self.arity != 1 or other.arity != 1:
            raise ValueError("composition is only implemented for unary operators")
        signature=None
        if self.signature is not None and other.signature is not None:
            if other.signature.arity!=1 or self.signature.arity!=1: raise ValueError("composition is only implemented for unary operator signatures")
            intermediate=other.signature.output_for(*other.signature.inputs)
            self.signature.validate_inputs(intermediate)
            def output_rule(F): return self.signature.output_for(other.signature.output_for(F))
            signature=OperatorSignature(inputs=other.signature.inputs,output_rule=output_rule,
                                        free_index_variance=other.index_variance+self.index_variance)
        return TensorOperator(index_variance=other.index_variance+self.index_variance,action=lambda field: self(other(field)),
                              name=f"{self.name}∘{other.name}",signature=signature,operator_kind="composition",metadata={"outer": self.name, "inner": other.name},)
    
    def raise_lower_index(self,index=0):
        """Raise or lower one free operator index using the metric.
        Parameters:
         - index : int, optional
            Position of the free operator index to raise or lower.
        Returns:
         - TensorOperator
            Operator with the selected free index variance reversed.
        """
        if not isinstance(index,int): raise TypeError("operator index must be an integer")
        if index<0 or index>=self.rank: raise IndexError("operator index out of range")
        if hasattr(self,"manifold"):
            from tensorium.core.manifold import MetricManifold
            if not isinstance(self.manifold, MetricManifold): raise ValueError("operator indices can only be raised or lowered on a MetricManifold")
        new_variance=list(self.index_variance)
        old_variance=new_variance[index]
        new_variance[index]*=-1
        new_variance=tuple(new_variance)
        new_index_variance=new_variance[index]
        def action(field):
            result=self(field)
            result_index=field.rank+index
            return result.raise_lower_indices((result_index,))
        signature=None
        if self.signature is not None:
            def output_rule(*input_signatures):
                output=self.signature.output_for(*input_signatures)
                dr=(1 if new_index_variance==1 else 0)-(1 if old_variance==1 else 0)
                ds=(1 if new_index_variance==-1 else 0)-(1 if old_variance==-1 else 0)
                shifted=output.with_tensor_type_shift(dr,ds)
                if output.index_variance is None: return shifted
                base_rank=len(output.index_variance)-self.rank
                if base_rank<0: raise ValueError("operator signature has fewer geometric indices than expected")
                result_index=base_rank+index
                index_variance=list(output.index_variance)
                index_variance[result_index]=new_index_variance
                return FieldSignature(tensor_type=shifted.tensor_type,index_variance=tuple(index_variance),
                                      internal_shape=output.internal_shape,internal_variance=output.internal_variance,
                                      manifold=output.manifold,generic=output.generic)
            signature=OperatorSignature(inputs=self.signature.inputs,output_rule=output_rule,free_index_variance=new_variance)
        return TensorOperator(index_variance=new_variance,action=action,name=self.name,signature=signature,operator_kind="raise_lower",metadata={"source": self.name, "index": index},)

    def raise_index(self,index=0):
        """Raise a covariant free operator index.
        Parameters:
         - index : int, optional
            Position of the covariant free index.
        Returns:
         - TensorOperator
            Operator with the selected index raised.
        """
        if self.index_variance[index]!=-1: raise ValueError("only covariant indices can be raised")
        return self.raise_lower_index(index)

    def lower_index(self,index=0):
        """Lower a contravariant free operator index.
        Parameters:
         - index : int, optional
            Position of the contravariant free index.
        Returns:
         - TensorOperator
            Operator with the selected index lowered.
        """
        if self.index_variance[index]!=1: raise ValueError("only contravariant indices can be lowered")
        return self.raise_lower_index(index)

    def contract_with(self,other,self_index=0,other_index=0):
        """Contract a free index of this operator with one of another operator.
        Parameters:
         - other : TensorOperator
            Second unary operator.
         - self_index : int, optional
            Free index of ``self`` to contract.
         - other_index : int, optional
            Free index of ``other`` to contract.
        Returns:
         - TensorOperator
            Composite operator with the selected pair of free indices
            contracted.
        """
        if not isinstance(other,TensorOperator): raise ValueError("can only contract with another TensorOperator")
        if not isinstance(self_index,int) or not isinstance(other_index,int): raise TypeError("operator indices must be integers")
        if other_index<0 or other_index>=other.rank: raise ValueError("other_index out of range")
        if self_index<0 or self_index>=self.rank: raise ValueError("self_index out of range")
        if self.arity != 1 or other.arity != 1: raise ValueError("contract_with is only implemented for unary operators")
        self_variance=self.index_variance[self_index]
        other_variance=other.index_variance[other_index]
        if self_variance==other_variance: raise ValueError("contracted operator indices must have opposite variance")
        combined_variance=other.index_variance+self.index_variance
        other_pos_in_combined=other_index
        self_pos_in_combined=other.rank+self_index
        new_index_variance=tuple(variance for pos,variance in enumerate(combined_variance) if pos not in (other_pos_in_combined,self_pos_in_combined))
        def action(field):
            result=self(other(field))
            other_result_pos=field.rank+other_index
            self_result_pos=field.rank+other.rank+self_index
            if other_variance==1:
                contravariant_pos=other_result_pos
                covariant_pos=self_result_pos
            else:
                contravariant_pos=self_result_pos
                covariant_pos=other_result_pos
            return result.contraction(contravariant_pos, covariant_pos)
        signature=None
        if self.signature is not None and other.signature is not None:
            if self.signature is not None and other.signature is not None:
                if self.signature.arity!=1 or other.signature.arity!=1: raise ValueError("contract_with is only implemented for unary operator signatures")
            def output_rule(F):
                after_other=other.signature.output_for(F)
                after_self=self.signature.output_for(after_other)
                if after_self.index_variance is None: return after_self.without_geometric_indices((other_pos_in_combined,self_pos_in_combined))
                base_rank=len(after_self.index_variance)-len(combined_variance)
                if base_rank<0: raise ValueError("operator signature has fewer geometric indices than expected")
                other_pos=base_rank+other_pos_in_combined
                self_pos=base_rank+self_pos_in_combined
                if after_self.index_variance[other_pos]==after_self.index_variance[self_pos]: raise ValueError("contracted signature indices must have opposite variance")
                return after_self.without_geometric_indices((other_pos,self_pos))
            test_output=output_rule(*other.signature.inputs)
            signature=OperatorSignature(inputs=other.signature.inputs,output_rule=output_rule,free_index_variance=new_index_variance)

        return TensorOperator(index_variance=new_index_variance,action=action,name=f"{self.name}·{other.name}",signature=signature,operator_kind="operator_contraction",
                              metadata={"left": self.name, "right": other.name, "self_index": self_index, "other_index": other_index},)

    @staticmethod
    def identity_operator(name="Id"):
        """Return the identity operator on tensor fields.
        Parameters:
         - name : str, optional
            Symbolic name of the operator.
        Returns:
         - TensorOperator
            Unary scalar operator satisfying ``Id(T)=T``.
        """
        field=FieldSignature.generic_tensor()
        signature=OperatorSignature(inputs=(field,),output_rule=lambda F: F,free_index_variance=())
        return TensorOperator(index_variance=(), arity=1, action=lambda field: field, name=name, signature=signature,operator_kind="identity")
    @staticmethod
    def tensor_product_operator(name="⊗"):
        """Return the tensor-product operator.
        Parameters:
         - name : str, optional
            Symbolic name of the operator.
        Returns:
         - TensorOperator
            Binary scalar operator sending ``(A,B)`` to ``A⊗B``.
        """
        A=FieldSignature(tensor_type=("r","s"),internal_shape=None,internal_variance=None,generic=True)
        B=FieldSignature(tensor_type=("p","q"),internal_shape=None,internal_variance=None,generic=True)
        def output_rule(F,G):
            internal_shape=None if F.internal_shape is None or G.internal_shape is None else F.internal_shape+G.internal_shape
            internal_variance=None if F.internal_variance is None or G.internal_variance is None else F.internal_variance+G.internal_variance
            return FieldSignature(tensor_type=(f"{F.tensor_type[0]}+{G.tensor_type[0]}",f"{F.tensor_type[1]}+{G.tensor_type[1]}"),
                                  internal_shape=internal_shape,internal_variance=internal_variance,manifold=F.manifold,generic=True)
        signature=OperatorSignature(inputs=(A,B),output_rule=output_rule,free_index_variance=())
        return TensorOperator(index_variance=(),arity=2,action=lambda A,B: A.tensor_product(B),name=name,signature=signature,operator_kind="tensor_product",)
    @staticmethod
    def contraction_operator(i,j,name="Tr"):
        """Return an operator contracting two geometric indices.
        Parameters:
         - i : int
            Contravariant index position.
         - j : int
            Covariant index position.
         - name : str, optional
            Symbolic name of the operator.
        Returns:
         - TensorOperator
            Unary scalar operator applying ``T.contraction(i,j)``.
        """
        field=FieldSignature.generic_tensor()
        signature=OperatorSignature(inputs=(field,),output_rule=lambda F: F.with_tensor_type_shift(-1,-1),free_index_variance=())
        return TensorOperator(index_variance=(),arity=1, action=lambda T: T.contraction(i, j),name=name,signature=signature,operator_kind="contraction",metadata={"i": i, "j": j},)
    @staticmethod
    def multiplication_operator(multiplier,name=None):
        """Return multiplication by a fixed object.
        Parameters:
         - multiplier : TensorField, ValuedTensorField, or sympifiable object
            Object placed on the left of the argument.
         - name : str, optional
            Symbolic name of the operator.
        Returns:
         - TensorOperator
            Unary scalar operator sending ``T`` to ``T*multiplier``.
        """
        field=FieldSignature(tensor_type=("r","s"),internal_shape=None,internal_variance=None,generic=True)
        signature=OperatorSignature(inputs=(field,),output_rule=lambda F: F,free_index_variance=(),)
        return TensorOperator(index_variance=(),arity=1,action=lambda field: field*multiplier,name=name or f"M_{getattr(multiplier, 'name', 'f')}",
                              operator_kind="multiplication",signature=signature,metadata={"multiplier": getattr(multiplier, "name", None)},)

    def __add__(self,other):
        """Add compatible tensor operators.
        Parameters:
         - other : TensorOperator
            Operator with the same arity and free index variance.
        Returns:
         - TensorOperator
            Operator whose action is the sum of both actions.
        """
        if not isinstance(other,TensorOperator): return NotImplemented
        if self.arity!=other.arity: raise ValueError("can only add operators with the same arity")
        if self.index_variance!=other.index_variance: raise ValueError("can only add operators with the same free index variance")

        signature=self.signature
        if self.signature is not None and other.signature is not None:
            for pos,(left,right) in enumerate(zip(self.signature.inputs,other.signature.inputs)):
                if not left.compatible_with(right): raise ValueError(f"operator inputs at position {pos} are incompatible")
            left_outputs=[self.signature.output_for(*self.signature.inputs),self.signature.output_for(*other.signature.inputs)]
            right_outputs=[other.signature.output_for(*self.signature.inputs),other.signature.output_for(*other.signature.inputs)]
            for left_output,right_output in zip(left_outputs,right_outputs):
                if not left_output.compatible_with(right_output): raise ValueError("operator outputs are incompatible")
            signature=OperatorSignature(inputs=self.signature.inputs,output_rule=self.signature.output_rule,free_index_variance=self.index_variance)

        def action(*args): return self(*args)+other(*args)
        return TensorOperator(index_variance=self.index_variance,action=action,name=f"{self.name}+{other.name}",
                              arity=self.arity,signature=signature,operator_kind="sum",metadata={"left": self.name, "right": other.name})

    def __neg__(self):
        """Return the additive inverse operator.
        Returns:
         - TensorOperator
            Operator sending each input to minus the original output.
        """
        return TensorOperator(index_variance=self.index_variance,action=lambda *args:-self(*args),name=f"-{self.name}",arity=self.arity,signature=self.signature,
                              operator_kind="negation",metadata={"source": self.name})
    def __sub__(self,other):
        """Subtract compatible tensor operators.
        Parameters:
         - other : TensorOperator
            Operator to subtract.
        Returns:
         - TensorOperator
            Difference operator.
        """
        return self+(-other)

    def __mul__(self,scalar):
        """Multiply the output of the operator by a scalar-like object.
        Parameters:
         - scalar : object
            Right scalar-like factor applied to the output.
        Returns:
         - TensorOperator
            Operator with scaled output.
        """
        if isinstance(scalar,TensorOperator): return NotImplemented
        return TensorOperator(index_variance=self.index_variance,action=lambda *args: self(*args)*scalar,name=f"{self.name}",arity=self.arity,signature=self.signature,
                              operator_kind="scaled",metadata={"source": self.name})
    def __rmul__(self,scalar):
        """Multiply the output of the operator by a scalar-like object on the left.
        Parameters:
         - scalar : object
            Left scalar-like factor applied to the output.
        Returns:
         - TensorOperator
            Operator with scaled output.
        """
        if isinstance(scalar,TensorOperator): return NotImplemented
        return TensorOperator(index_variance=self.index_variance,action=lambda *args: scalar*self(*args),name=f"{self.name}",arity=self.arity,signature=self.signature,
                              operator_kind="scaled",metadata={"source": self.name})

    def commutator(self,other,name=None):
        """Return the commutator with another unary operator.
        Parameters:
         - other : TensorOperator
            Unary operator to commute with ``self``.
         - name : str, optional
            Symbolic name of the commutator.
        Returns:
         - TensorOperator
            Operator ``self∘other - other∘self``.
        """
        if not isinstance(other,TensorOperator): raise ValueError("Other must be a TensorOperator")
        if self.arity!=1 or other.arity!=1: raise ValueError("commutator is only implemented for unary operators")
        first=self.compose(other)
        second=other.compose(self)
        if first.index_variance!=second.index_variance: raise ValueError("commutator terms have incompatible free index variance")
        commutator_name=name or f"[{self.name},{other.name}]"
        return TensorOperator(index_variance=first.index_variance,action=lambda field: first(field)-second(field),name=commutator_name,arity=1,signature=first.signature,
                              operator_kind="commutator",metadata={"left": self.name, "right": other.name},)

def commutator(A,B,name=None):
    """Return the commutator of two tensor operators.
    Parameters:
     - A : TensorOperator
        First unary operator.
     - B : TensorOperator
        Second unary operator.
     - name : str, optional
        Symbolic name of the commutator.
    Returns:
     - TensorOperator
        Operator ``A∘B - B∘A``.
    """
    if not isinstance(A,TensorOperator) or not isinstance(B,TensorOperator):
        raise ValueError("A and B must both be TensorOperator")
    return A.commutator(B,name=name)

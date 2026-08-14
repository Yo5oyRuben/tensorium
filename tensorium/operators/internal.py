from tensorium.operators.base import TensorOperator
from tensorium.fields import ValuedTensorField
from tensorium.operators.signature import FieldSignature, OperatorSignature

def internal_action_operator(matrix,index=0,name=None):
    """Return the operator induced by an internal matrix-valued tensor field.
    Parameters:
     - matrix : ValuedTensorField
        Matrix-valued tensor field with internal variance ``(1,-1)``. Its free
        geometric indices become free indices of the resulting operator.
     - index : int, optional
        Internal index of the argument on which the matrix acts.
     - name : str, optional
        Symbolic name of the operator.
    Returns:
     - TensorOperator
        Unary operator acting on the selected internal index of an
        ``ValuedTensorField``.
    """
    if not isinstance(matrix,ValuedTensorField): raise ValueError("matrix must be a ValuedTensorField")
    if matrix.internal_rank!=2: raise ValueError("matrix must have two internal indices")
    if matrix.internal_variance!=(1,-1): raise ValueError("matrix must have internal variance (1, -1)")
    if matrix.internal_shape[0]!=matrix.internal_shape[1]: raise ValueError("matrix must be square")
    def action(field):
        if not isinstance(field,ValuedTensorField): raise ValueError("internal action can only act on ValuedTensorField")
        if not isinstance(index,int): raise TypeError("internal index must be an integer")
        if index<0 or index>=field.internal_rank: raise IndexError("internal index out of range")
        if field.internal_shape[index]!=matrix.internal_shape[0]: raise ValueError("internal dimensions do not match")
        internal_variance=field.internal_variance
        if isinstance(internal_variance,int): internal_variance=(internal_variance,)
        variance=internal_variance[index]
        if variance==1: result=matrix.internal_action(field,pairs=((1,index),))
        elif variance==-1: result=matrix.internal_action(field,pairs=((0,index),))
        else: raise ValueError("internal variance must be 1 or -1")
        if matrix.rank:
            order=tuple(range(matrix.rank, matrix.rank+field.rank))+tuple(range(matrix.rank))
            result=result.permute_geometric_indices(order)
        order=tuple(range(1,index+1))+(0,)+tuple(range(index+1,field.internal_rank))
        return result.permute_internal_indices(order)
    field=FieldSignature.generic_internal_tensor()
    signature=OperatorSignature(inputs=(field,),output_rule=lambda F: F.with_extra_geometric_indices(matrix.index_variance),free_index_variance=matrix.index_variance)
    return TensorOperator(index_variance=matrix.index_variance,action=action,name=name or getattr(matrix,"name","A"),arity=1,signature=signature,
                          operator_kind="internal_action",metadata={"internal_shape": matrix.internal_shape, "internal_variance": matrix.internal_variance})

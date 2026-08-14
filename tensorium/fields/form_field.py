from tensorium.fields.tensor_field import TensorField

class OneForm(TensorField):
    """Represent a one-form field on a manifold.
    A ``OneForm`` is a specialized ``TensorField`` of type ``(0, 1)``.
    Its local data are stored as coordinate covector components on one or
    more charts.
    """
    def __init__(self, manifold, local_representations):
        """Create a one-form field from local representations.
        Parameters:
         - manifold : Manifold (Manifold on which the one-form is defined)
         - local_representations : dict
            Dictionary mapping charts to local tensor representations of the
            one-form.
        Notes:
        A one-form is always treated as a tensor of type ``(0, 1)`` with index
        variance ``(-1,)``.
        """
        super().__init__(manifold, (0,1), local_representations, (-1,))
     
    def __call__(self, vector):
        """Apply the one-form to a vector field.
        Parameters:
         - vector : TensorField
            Vector field on which the one-form is evaluated. It must have tensor
            type ``(1, 0)``.
        Returns:
         - TensorField: A rank-zero tensor field representing the resulting scalar field.
        Notes:
        This method delegates the actual evaluation to the generic tensor field
        call logic after validating the argument type.
        """
        if not isinstance(vector, TensorField): raise ValueError("vector must be a Vector, thus, it must be a TensorField")
        if not vector.tensor_type==(1,0): raise ValueError("A OneForm must act on a Vector field")
        return super().__call__(vector)
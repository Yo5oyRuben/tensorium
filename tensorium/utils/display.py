from itertools import product

from sympy import Matrix, cancel, factor, latex, pretty

DISPLAY_MAX_LATEX_CHARS = 25000
DISPLAY_MAX_ENTRY_CHARS = 1200
DISPLAY_INLINE_CHARS = 220
DISPLAY_MAX_BLOCK_LINES = 80


def _maybe_display_latex(expr):
    """Display LaTeX in IPython when available, otherwise print it."""
    expr_length = len(str(expr))
    if expr_length > DISPLAY_MAX_LATEX_CHARS:
        expr = (
            r"\begin{gathered}"
            r"\text{Display output omitted: expression too large.}\\"
            rf"\text{{Output length: {expr_length} characters.}}"
            r"\end{gathered}"
        )
    try:
        from IPython.display import Math, display
    except ImportError:
        print(expr)
        return
    display(Math(expr))


def _maybe_display_text(text):
    """Display plain text in IPython when available, otherwise print it."""
    try:
        from IPython.display import display
    except ImportError:
        print(text)
        return
    display(text)


def _tensor_index_latex(index_variance, index_values=None):
    """Build the LaTeX superscript/subscript pattern for tensor indices."""
    if index_values is None:
        index_values = [str(i) for i in range(len(index_variance))]
    upper = [str(v) for slot, v in enumerate(index_values) if index_variance[slot] == 1]
    lower = [str(v) for slot, v in enumerate(index_values) if index_variance[slot] == -1]

    pieces = []
    if upper:
        pieces.append("^{" + " ".join(upper) + "}")
    if lower:
        pieces.append("_{" + " ".join(lower) + "}")
    return "".join(pieces)


def _tensor_name_latex(name, index_variance, index_values=None):
    """Build a tensor name with LaTeX indices."""
    base = "T" if name is None else str(name)
    if ("^" in base or "_" in base) and "[" not in base:
        if index_values is None:
            base = "{" + base + "}"
        else:
            base = r"\left(" + base + r"\right)"
    return base + _tensor_index_latex(index_variance, index_values=index_values)


def _combined_tensor_name_latex(name, internal_variance, internal_values, index_variance, index_values=None):
    """Build a tensor name combining internal and geometric indices."""
    base = _plain_tensor_name_latex(name)
    if isinstance(internal_variance, int):
        internal_variance = (internal_variance,)
    if index_values is None:
        index_values = [str(i) for i in range(len(index_variance))]
    upper = [rf"[{value}]" for slot, value in enumerate(internal_values) if internal_variance[slot] == 1]
    lower = [rf"[{value}]" for slot, value in enumerate(internal_values) if internal_variance[slot] == -1]
    upper.extend(str(value) for slot, value in enumerate(index_values) if index_variance[slot] == 1)
    lower.extend(str(value) for slot, value in enumerate(index_values) if index_variance[slot] == -1)

    pieces = []
    if upper:
        pieces.append("^{" + " ".join(upper) + "}")
    if lower:
        pieces.append("_{" + " ".join(lower) + "}")
    return base + "".join(pieces)


def _internal_index_latex(internal_variance, index_values=None):
    """Build bracketed indices for finite-dimensional internal slots."""
    if isinstance(internal_variance, int):
        internal_variance = (internal_variance,)
    if index_values is None:
        index_values = [str(i) for i in range(len(internal_variance))]
    upper = [rf"[{v}]" for slot, v in enumerate(index_values) if internal_variance[slot] == 1]
    lower = [rf"[{v}]" for slot, v in enumerate(index_values) if internal_variance[slot] == -1]

    pieces = []
    if upper:
        pieces.append("^{" + " ".join(upper) + "}")
    if lower:
        pieces.append("_{" + " ".join(lower) + "}")
    return "".join(pieces)


def _internal_index_pattern_description(internal_variance):
    """Describe the internal index pattern without choosing components."""
    if isinstance(internal_variance, int):
        internal_variance = (internal_variance,)
    upper = sum(1 for variance in internal_variance if variance == 1)
    lower = sum(1 for variance in internal_variance if variance == -1)
    pieces = []
    if upper:
        pieces.append(rf"{upper}\ \text{{upper}}")
    if lower:
        pieces.append(rf"{lower}\ \text{{lower}}")
    if not pieces:
        return r"\text{none}"
    return r",\ ".join(pieces)


def _plain_tensor_name_latex(name=None):
    """Build a tensor name without component indices."""
    base = "T" if name is None else str(name)
    if "^" in base or "_" in base:
        base = "{" + base + "}"
    return base


def _plain_object_name_latex(name):
    """Return a readable symbolic name without SymPy string formatting."""
    return str(name)


def _manifold_name_latex(obj):
    """Return the manifold name attached to a tensor-like object."""
    if hasattr(obj, "base_manifold"):
        return _plain_object_name_latex(obj.base_manifold.name)
    if hasattr(obj, "manifold"):
        return _plain_object_name_latex(obj.manifold.name)
    if hasattr(obj, "chart") and hasattr(obj.chart, "manifold"):
        return _plain_object_name_latex(obj.chart.manifold.name)
    return "M"


def _field_module_latex(tensor_type, manifold):
    """Return the mathematical space where a tensor field lives."""
    if tuple(tensor_type) == (0, 0):
        return rf"C^\infty({manifold})"
    return _tensor_module_latex(tensor_type[0], tensor_type[1], manifold)


def _internal_space_factor_latex(dim, variance):
    """Return the internal vector-space factor for one internal index."""
    if variance == 1:
        return rf"E_{{{dim}}}"
    return rf"E_{{{dim}}}^*"


def _internal_factors_latex(internal_shape, internal_variance):
    """Return all internal vector-space factors with their dimensions."""
    factors = [
        _internal_space_factor_latex(dim, variance)
        for dim, variance in zip(internal_shape, internal_variance)
    ]
    if not factors:
        return ""
    return r"\otimes " + r"\otimes ".join(factors)


def _field_membership_latex(label, tensor_type, manifold, internal_shape=(), internal_variance=()):
    """Return a compact membership statement for tensor-like fields."""
    module = _field_module_latex(tensor_type, manifold)
    module += _internal_factors_latex(tuple(internal_shape), tuple(internal_variance))
    return rf"{_plain_tensor_name_latex(label)}\in {module}"


def _display_expr_latex(expr):
    """Return a lightly normalized LaTeX representation for display only."""
    raw_length = len(str(expr))
    if raw_length > DISPLAY_MAX_ENTRY_CHARS:
        return rf"\text{{large expression omitted ({raw_length} chars)}}"
    try:
        rendered = latex(factor(cancel(expr)))
    except Exception:
        rendered = latex(expr)
    if len(rendered) > DISPLAY_MAX_ENTRY_CHARS:
        return rf"\text{{large expression omitted ({len(rendered)} chars)}}"
    return rendered


def _display_matrix_latex(matrix):
    """Return LaTeX for a matrix after lightweight entry-wise normalization."""
    raw_length = sum(len(str(entry)) for entry in matrix)
    if raw_length > DISPLAY_MAX_LATEX_CHARS:
        return rf"\left[\text{{large matrix omitted ({raw_length} chars)}}\right]"
    try:
        matrix = matrix.applyfunc(
            lambda expr: expr if len(str(expr)) > DISPLAY_MAX_ENTRY_CHARS else factor(cancel(expr))
        )
    except Exception:
        pass
    rendered = latex(matrix)
    if len(rendered) > DISPLAY_MAX_ENTRY_CHARS:
        return rf"\left[\text{{large matrix omitted ({len(rendered)} chars)}}\right]"
    return rendered


def _matrix_latex_is_omitted(rendered):
    """Return True when a matrix display was replaced by an omission marker."""
    return r"\text{large matrix omitted" in rendered


def _operator_name_latex(name=None):
    """Build a LaTeX-safe operator name."""
    if name is None:
        return r"\mathcal{O}"
    label = str(name)
    label = label.replace("∇", r"\nabla")
    label = label.replace("·", r"\cdot")
    label = label.replace("∘", r"\circ")
    return label


def _operator_free_index_latex(index_variance):
    """Build symbolic free indices for operators."""
    names = (r"\mu", r"\nu", r"\rho", r"\sigma", r"\alpha", r"\beta")
    index_values = [names[i] if i < len(names) else f"i_{i}" for i in range(len(index_variance))]
    return _tensor_index_latex(index_variance, index_values=index_values)


def _operator_label_has_explicit_free_indices(label):
    """Return whether an operator label already contains symbolic free indices."""
    index_tokens = (r"\mu", r"\nu", r"\rho", r"\sigma", r"\alpha", r"\beta")
    return any(token in label for token in index_tokens)


def _operator_type_symbols(index_variance):
    """Return symbolic output shifts induced by free operator indices."""
    up = sum(1 for variance in index_variance if variance == 1)
    down = sum(1 for variance in index_variance if variance == -1)
    r_out = "r" if up == 0 else "r+1" if up == 1 else f"r+{up}"
    s_out = "s" if down == 0 else "s+1" if down == 1 else f"s+{down}"
    return r_out, s_out


def _symbol_plus(base, shift):
    """Return a compact symbolic expression ``base+shift``."""
    if shift == 0: return base
    if shift > 0: return f"{base}+{shift}" if shift > 1 else f"{base}+1"
    return f"{base}{shift}" if shift < -1 else f"{base}-1"


def _tensor_module_latex(r="r", s="s", manifold="M", internal=None):
    """Return a compact tensor-module notation."""
    module = rf"\mathcal{{T}}^{{{r}}}_{{{s}}}({_plain_object_name_latex(manifold)})"
    if internal:
        module += internal
    return module


def _internal_module_factor_latex(upper="a", lower="b"):
    """Return the generic internal vector-space factor."""
    upper_factor = rf"E^{{\otimes {upper}}}"
    lower_factor = rf"(E^*)^{{\otimes {lower}}}"
    return rf"\otimes {upper_factor}\otimes {lower_factor}"


def _field_signature_module_latex(signature, fallback_manifold="M"):
    """Return module notation associated with a ``FieldSignature``."""
    manifold = getattr(signature, "manifold", None)
    manifold = fallback_manifold if manifold is None else getattr(manifold, "name", manifold)
    tensor_type = getattr(signature, "tensor_type", None)
    if tensor_type is None:
        module = r"\text{fields}"
    else:
        module = _field_module_latex(tensor_type, _plain_object_name_latex(manifold))
    internal_shape = getattr(signature, "internal_shape", ())
    internal_variance = getattr(signature, "internal_variance", ())
    internal_shape = () if internal_shape is None else tuple(internal_shape)
    internal_variance = () if internal_variance is None else tuple(internal_variance)
    if internal_shape:
        if getattr(signature, "generic", False):
            upper = [str(dim) for dim, variance in zip(internal_shape, internal_variance) if variance == 1]
            lower = [str(dim) for dim, variance in zip(internal_shape, internal_variance) if variance == -1]
            if upper:
                module += r"\otimes " + r"\otimes ".join(rf"E^{{\otimes {dim}}}" for dim in upper)
            if lower:
                module += r"\otimes " + r"\otimes ".join(rf"(E^*)^{{\otimes {dim}}}" for dim in lower)
        else:
            module += _internal_factors_latex(internal_shape, internal_variance)
    return module


def _operator_signature_line_latex(operator, label, free_indices, manifold):
    """Return the signature line of an operator when metadata is available."""
    signature = getattr(operator, "signature", None)
    if signature is None: return None
    inputs = tuple(signature.inputs)
    output = signature.output_for(*inputs)
    input_latex = r"\times ".join(_field_signature_module_latex(F, manifold) for F in inputs)
    output_latex = _field_signature_module_latex(output, manifold)
    return rf"{label}{free_indices}: {input_latex}\longrightarrow {output_latex}"


def _internal_matrix_module_latex(r="r", s="s", manifold="M"):
    """Return notation for matrix-valued tensor fields."""
    tensor_module = _tensor_module_latex(r, s, manifold)
    return rf"{tensor_module}\otimes E\otimes E^* \simeq {tensor_module}\otimes \operatorname{{End}}(E)"


def _local_tensor_expression_latex_lines(local_tensor, name=None, name_builder=None):
    """Return compact LaTeX lines without chart metadata."""
    if name_builder is None:
        name_builder = lambda index_values=None: _tensor_name_latex(
            name, local_tensor.index_variance, index_values
        )
    tensor_name = name_builder()
    if local_tensor.rank == 0:
        return [rf"{tensor_name} = {_display_expr_latex(local_tensor.components[()])}"]

    if local_tensor.rank == 1:
        mat = Matrix(list(local_tensor.components))
        vector_index = [r"\mu"]
        vector_name = name_builder(vector_index)
        return [rf"{vector_name} = {_display_matrix_latex(mat)}"]

    if local_tensor.rank == 2:
        try:
            mat = Matrix(local_tensor.to_matrix())
            matrix_indices = [r"\mu", r"\nu"]
            matrix_name = name_builder(matrix_indices)
            return [rf"{matrix_name} = {_display_matrix_latex(mat)}"]
        except Exception:
            pass

    grouped_lines = _local_tensor_block_matrix_latex_lines(local_tensor, name=name, name_builder=name_builder)
    if grouped_lines is not None:
        return grouped_lines

    lines = []
    dim = local_tensor.chart.dim
    total_components = dim**local_tensor.rank
    for count, index in enumerate(product(range(dim), repeat=local_tensor.rank)):
        if count >= DISPLAY_MAX_BLOCK_LINES:
            remaining = total_components - DISPLAY_MAX_BLOCK_LINES
            lines.append(rf"\text{{{remaining} further components omitted}}")
            break
        indexed_name = name_builder(index)
        lines.append(rf"{indexed_name} = {_display_expr_latex(local_tensor[index])}")
    return lines


def _local_tensor_block_matrix_latex_lines(local_tensor, name=None, name_builder=None):
    """Group tensors of rank > 2 as matrices in the last two indices."""
    if name_builder is None:
        name_builder = lambda index_values=None: _tensor_name_latex(
            name, local_tensor.index_variance, index_values
        )
    if local_tensor.rank <= 2:
        return None
    dim = local_tensor.chart.dim
    leading_rank = local_tensor.rank - 2
    leading_indices = list(product(range(dim), repeat=leading_rank))
    if len(leading_indices) > DISPLAY_MAX_BLOCK_LINES:
        return None

    lines = []
    symbolic_tail = [r"\mu", r"\nu"]
    for fixed in leading_indices:
        matrix = Matrix([
            [local_tensor[fixed + (i, j)] for j in range(dim)]
            for i in range(dim)
        ])
        rendered_matrix = _display_matrix_latex(matrix)
        if _matrix_latex_is_omitted(rendered_matrix):
            return None
        index_values = list(fixed) + symbolic_tail
        indexed_name = name_builder(index_values)
        lines.append(rf"{indexed_name} = {rendered_matrix}")
    return lines


def _local_tensor_compact_latex_lines(local_tensor, name=None):
    """Return compact LaTeX lines for a local tensor representation."""
    prefix = (
        rf"{latex(local_tensor.chart.name)}"
        rf"\text{{ on }} {latex(local_tensor.open_set.name)}:\quad "
    )

    expr_lines = _local_tensor_expression_latex_lines(local_tensor, name=name)
    if local_tensor.rank <= 2:
        return [prefix + expr_lines[0]]

    lines = [prefix]
    for expr in expr_lines:
        lines.append(rf"\qquad {expr}")
    return lines


def tensor_component_dict(local_tensor):
    """Return all tensor components as a dictionary indexed by tuples."""
    data = {}
    dim = local_tensor.chart.dim
    for index in product(range(dim), repeat=local_tensor.rank):
        data[index] = local_tensor[index]
    return data


def print_local_tensor(local_tensor, name=None):
    """Print a compact text summary of a local tensor."""
    label = "Tensor" if name is None else name
    print(f"{label} on chart {local_tensor.chart.name}")
    print(f"  type = {local_tensor.tensor_type}")
    print(f"  index_variance = {local_tensor.index_variance}")
    print(f"  rank = {local_tensor.rank}")
    print(f"  components = {local_tensor.components}")


def print_tensor_in_chart(tensor, chart, name=None):
    """Print a tensor field after expressing it in a chosen chart."""
    local_tensor = tensor.local_representation(chart)
    print_local_tensor(local_tensor, name=name)


def display_local_tensor(local_tensor, name=None):
    """Display a local tensor using LaTeX when possible."""
    tensor_name = _tensor_name_latex(name, local_tensor.index_variance)
    label = "T" if name is None else str(name)
    membership = _field_membership_latex(label, local_tensor.tensor_type, _manifold_name_latex(local_tensor))
    chart_prefix = (
        rf"{membership},\quad "
        rf"{latex(local_tensor.chart.name)}\text{{ on }} {latex(local_tensor.open_set.name)}"
    )
    if local_tensor.rank == 0:
        _maybe_display_latex(
            rf"{chart_prefix}:\quad {tensor_name} = {_display_expr_latex(local_tensor.components[()])}"
        )
        return

    _maybe_display_latex(chart_prefix)

    if local_tensor.rank == 1:
        mat = Matrix(list(local_tensor.components))
        vector_index = [r"\mu"]
        vector_name = _tensor_name_latex(name, local_tensor.index_variance, vector_index)
        _maybe_display_latex(rf"{vector_name} = {_display_matrix_latex(mat)}")
        return

    if local_tensor.rank == 2:
        try:
            mat = Matrix(local_tensor.to_matrix())
            matrix_indices = [r"\mu", r"\nu"]
            matrix_name = _tensor_name_latex(name, local_tensor.index_variance, matrix_indices)
            _maybe_display_latex(rf"{matrix_name} = {_display_matrix_latex(mat)}")
            return
        except Exception:
            pass

    grouped_lines = _local_tensor_block_matrix_latex_lines(local_tensor, name=name)
    if grouped_lines is not None:
        _maybe_display_latex(r"\\ ".join(grouped_lines))
        return

    rows = []
    dim = local_tensor.chart.dim
    total_components = dim**local_tensor.rank
    for count, index in enumerate(product(range(dim), repeat=local_tensor.rank)):
        if count >= DISPLAY_MAX_BLOCK_LINES:
            remaining = total_components - DISPLAY_MAX_BLOCK_LINES
            rows.append(rf"\text{{{remaining} further components omitted}}")
            break
        indexed_name = _tensor_name_latex(name, local_tensor.index_variance, index)
        rows.append(rf"{indexed_name} = {_display_expr_latex(local_tensor[index])}")
    _maybe_display_latex(r"\\ ".join(rows))


def display_tensor_in_chart(tensor, chart, name=None):
    """Display a tensor field in a specific chart."""
    local_tensor = tensor.local_representation(chart)
    display_local_tensor(local_tensor, name=name)


def display_tensor_field(tensor, chart=None, name=None):
    """Display one chart or all known local representations of a tensor field."""
    if len(tensor.local_representations) == 0:
        raise ValueError("The tensor field has no known local representations.")
    if chart is not None:
        return display_tensor_in_chart(tensor, chart, name=name)

    label = "T" if name is None else str(name)
    lines = [
        _field_membership_latex(label, tensor.tensor_type, _manifold_name_latex(tensor))
    ]
    for local_tensor in tensor.local_representations.values():
        lines.extend(_local_tensor_compact_latex_lines(local_tensor, name=name))
    _maybe_display_latex(r"\begin{gathered}" + r"\\[0.35em]".join(lines) + r"\end{gathered}")


def _internal_component_items(components, prefix=()):
    """Yield ``(internal_index, TensorField)`` pairs from nested components."""
    if _is_tensor_field_like(components):
        yield prefix, components
        return
    for index, component in enumerate(components):
        yield from _internal_component_items(component, prefix + (index,))


def _internal_symbolic_index_values(internal_variance):
    """Return symbolic labels for internal slots, separating upper and lower ones."""
    if isinstance(internal_variance, int):
        internal_variance = (internal_variance,)
    upper_names = iter(("A", "B", "C", "D", "E"))
    lower_names = iter(("I", "J", "K", "L", "M"))
    values = []
    for variance in internal_variance:
        values.append(next(upper_names) if variance == 1 else next(lower_names))
    return values


def _internal_symbolic_name_latex(name, internal_variance):
    """Return a symbolic internal-index name such as Q^{[A]}_{[I]}."""
    return (
        _plain_tensor_name_latex(name)
        + _internal_index_latex(internal_variance, _internal_symbolic_index_values(internal_variance))
    )


def _local_scalar_expr_for_component(component, local_chart, transform=False):
    """Return a scalar component in a chart, or None when compact display is unsuitable."""
    if transform:
        local_tensor = component.local_representation(local_chart)
    else:
        local_tensor = component.local_representations.get(local_chart)
    if local_tensor is None or local_tensor.rank != 0:
        return None
    return local_tensor.components[()]


def _internal_scalar_matrix_latex(internal_tensor, local_chart, label, internal_variance, transform=False):
    """Return a matrix/vector display line for scalar-valued internal tensors."""
    if internal_tensor.tensor_type != (0, 0) or internal_tensor.internal_rank not in (1, 2):
        return None
    if internal_tensor.internal_rank == 1:
        entries = []
        for i in range(internal_tensor.internal_shape[0]):
            expr = _local_scalar_expr_for_component(internal_tensor[i], local_chart, transform=transform)
            if expr is None:
                return None
            entries.append(expr)
        matrix = Matrix(entries)
    else:
        rows = []
        for i in range(internal_tensor.internal_shape[0]):
            row = []
            for j in range(internal_tensor.internal_shape[1]):
                expr = _local_scalar_expr_for_component(internal_tensor[i, j], local_chart, transform=transform)
                if expr is None:
                    return None
                row.append(expr)
            rows.append(row)
        matrix = Matrix(rows)

    name = _internal_symbolic_name_latex(label, internal_variance)
    prefix = rf"{latex(local_chart.name)}\text{{ on }} {latex(local_chart.open_set.name)}:\quad "
    return rf"{prefix}{name} = {_display_matrix_latex(matrix)}"


def _is_tensor_field_like(obj):
    """Return True for global tensor fields, even if modules were reloaded."""
    return (
        hasattr(obj, "local_representations")
        and hasattr(obj, "local_representation")
        and hasattr(obj, "tensor_type")
        and hasattr(obj, "index_variance")
    )


def _is_internal_tensor_field_like(obj):
    """Return True for tensor fields carrying internal indices."""
    return (
        hasattr(obj, "components")
        and hasattr(obj, "internal_shape")
        and hasattr(obj, "internal_variance")
        and hasattr(obj, "internal_rank")
    )


def _is_connection_like(obj):
    """Return True for global connections, even if modules were reloaded."""
    return (
        hasattr(obj, "local_representations")
        and (hasattr(obj, "base_manifold") or hasattr(obj, "manifold"))
        and not _is_tensor_field_like(obj)
    )


def _is_tensor_operator_like(obj):
    """Return True for tensor operators, even if modules were reloaded."""
    return (
        hasattr(obj, "index_variance")
        and hasattr(obj, "tensor_type")
        and callable(obj)
    )


def _display_internal_tensor_field(internal_tensor, chart=None, name=None):
    """Display a tensor field carrying finite-dimensional internal indices."""
    label = "A" if name is None else str(name)
    internal_variance = internal_tensor.internal_variance
    if isinstance(internal_variance, int):
        internal_variance = (internal_variance,)
    lines = [
        _field_membership_latex(
            label,
            internal_tensor.tensor_type,
            _manifold_name_latex(internal_tensor),
            internal_tensor.internal_shape,
            internal_variance,
        )
    ]

    if chart is None:
        charts = []
        for _, component in _internal_component_items(internal_tensor.components):
            for local_chart in component.local_representations:
                if local_chart not in charts:
                    charts.append(local_chart)
    else:
        charts = [chart]

    for local_chart in charts:
        chart_lines = []
        open_set = local_chart.open_set
        compact_line = _internal_scalar_matrix_latex(
            internal_tensor, local_chart, label, internal_variance, transform=chart is not None
        )
        if compact_line is not None:
            lines.append(compact_line)
            continue
        for internal_index, component in _internal_component_items(internal_tensor.components):
            component_name = label + _internal_index_latex(internal_variance, internal_index)
            if chart is None:
                local_tensor = component.local_representations.get(local_chart)
                if local_tensor is None:
                    continue
            else:
                local_tensor = component.local_representation(local_chart)
            name_builder = lambda index_values=None, internal_index=internal_index, local_tensor=local_tensor: (
                _combined_tensor_name_latex(
                    label,
                    internal_variance,
                    internal_index,
                    local_tensor.index_variance,
                    index_values,
                )
            )
            chart_lines.extend(_local_tensor_expression_latex_lines(local_tensor, name_builder=name_builder))
        if chart_lines:
            lines.append(rf"{latex(local_chart.name)}\text{{ on }} {latex(open_set.name)}:")
            lines.extend(rf"\qquad {line}" for line in chart_lines)

    _maybe_display_latex(r"\begin{gathered}" + r"\\[0.35em]".join(lines) + r"\end{gathered}")


def pretty_local_tensor(local_tensor, name=None):
    """Return a pretty-printed string representation of a local tensor."""
    label = "Tensor" if name is None else name
    lines = [
        f"{label} on chart {local_tensor.chart.name}",
        f"type = {local_tensor.tensor_type}",
        f"index_variance = {local_tensor.index_variance}",
    ]

    if local_tensor.rank == 0:
        lines.append(pretty(local_tensor.components[()]))
        return "\n".join(lines)

    if local_tensor.rank == 2:
        try:
            lines.append(pretty(Matrix(local_tensor.to_matrix())))
            return "\n".join(lines)
        except Exception:
            pass

    for index, value in tensor_component_dict(local_tensor).items():
        lines.append(f"{index}: {pretty(value)}")
    return "\n".join(lines)


def _display_chart(chart):
    """Display basic information about a chart."""
    _maybe_display_latex(
        rf"\text{{Chart }} {latex(chart.name)}"
        rf"\text{{ on }} {latex(chart.manifold.name)}"
        rf",\quad \text{{open set}} = {latex(chart.open_set.name)}"
        rf",\quad \text{{symbols}} = {latex(tuple(chart.symbols))}"
    )


def _display_open_set(open_set):
    """Display basic information about an open set."""
    _maybe_display_latex(
        rf"\text{{Open set }} {latex(open_set.name)}"
        rf"\text{{ on }} {latex(open_set.manifold.name)}"
    )


def _display_manifold(manifold):
    """Display basic information about a manifold."""
    _maybe_display_latex(
        rf"\text{{Manifold }} {latex(manifold.name)}"
        rf",\quad \dim = {latex(manifold.dim)}"
    )


def _display_metric_manifold(metric_manifold):
    """Display basic information about a metric manifold."""
    _maybe_display_latex(
        rf"\text{{Metric manifold on }} {latex(metric_manifold.manifold.name)}"
    )


def _display_connection(connection, name=None):
    """Display basic information about a connection."""
    label = "Connection" if name is None else str(name)
    _maybe_display_latex(
        rf"{label}\text{{ on }} {latex(connection.base_manifold.name if hasattr(connection, 'base_manifold') else connection.manifold.name)}"
    )
    if hasattr(connection, "local_representations"):
        charts = ", ".join(str(chart.name) for chart in connection.local_representations)
        _maybe_display_text(f"Known local representations: {charts}")


def _display_local_connection(local_connection, name=None):
    """Display the components of a local connection."""
    label = r"\Gamma" if name is None else str(name)
    _maybe_display_latex(
        rf"{label}^{{\rho}}_{{\mu\nu}}"
        rf"\quad \text{{on chart }} {latex(local_connection.chart.name)}"
    )
    rows = []
    dim = local_connection.chart.dim
    for i, j, k in product(range(dim), repeat=3):
        rows.append(rf"{label}^{{{i}}}_{{{j}{k}}} = {latex(local_connection[(i, j, k)])}")
    _maybe_display_latex(r"\\ ".join(rows))


def _connection_summary_latex(connection):
    """Return a compact LaTeX description of a connection."""
    kind = connection.__class__.__name__
    manifold = connection.base_manifold.name if hasattr(connection, "base_manifold") else connection.manifold.name
    pieces = [rf"\text{{{kind} on }} {_plain_object_name_latex(manifold)}"]
    if hasattr(connection, "local_representations"):
        charts = ", ".join(str(chart.name) for chart in connection.local_representations)
        pieces.append(rf"\text{{charts}}=\{{{charts}\}}")
    if hasattr(connection, "internal_shape"):
        pieces.append(rf"\text{{internal shape}}={latex(connection.internal_shape)}")
    return r",\quad ".join(pieces)


def _operator_domain_latex(operator):
    """Describe the kind of fields accepted by an operator."""
    if hasattr(operator, "affine_connection") and hasattr(operator, "gauge_connection"):
        if operator.gauge_connection is not None:
            return r"\text{valued tensor fields}"
        return r"\text{tensor fields and valued tensor fields}"
    return r"\text{fields accepted by its stored action}"


def _display_tensor_operator(operator, name=None):
    """Display a tensor operator and the type of fields it acts on."""
    label = _operator_name_latex(name if name is not None else getattr(operator, "name", None))
    free_indices = _operator_free_index_latex(operator.index_variance)
    if free_indices and _operator_label_has_explicit_free_indices(label):
        free_indices = ""
    if free_indices and ("^" in label or "_" in label):
        label = "{" + label + "}"
    manifold = getattr(getattr(operator, "base_manifold", None), "name", "M")
    r_out, s_out = _operator_type_symbols(operator.index_variance)
    input_module = _tensor_module_latex("r", "s", manifold)
    output_module = _tensor_module_latex(r_out, s_out, manifold)
    kind = getattr(operator, "operator_kind", "generic")
    arity = getattr(operator, "arity", 1)
    lines = []
    signature_line = _operator_signature_line_latex(operator, label, free_indices, manifold)
    if signature_line is not None:
        lines.append(signature_line)
        if kind == "contraction":
            metadata = getattr(operator, "metadata", {})
            i = metadata.get("i", "?")
            j = metadata.get("j", "?")
            lines.append(rf"\text{{contracts geometric indices }} {i}\text{{ and }}{j}")
        _maybe_display_latex(r"\begin{aligned}" + r"\\[0.35em]".join(lines) + r"\end{aligned}")
        return

    if hasattr(operator, "affine_connection") and hasattr(operator, "gauge_connection"):
        if operator.gauge_connection is not None:
            internal = _internal_module_factor_latex("a", "b")
            lines.append(
                rf"{label}{free_indices}: "
                rf"{_tensor_module_latex('r', 's', manifold, internal)}"
                rf"\longrightarrow {_tensor_module_latex(r_out, s_out, manifold, internal)}"
            )
        else:
            lines.append(rf"{label}{free_indices}: {input_module}\longrightarrow {output_module}")
    elif kind == "tensor_product":
        left = _tensor_module_latex("r", "s", manifold)
        right = _tensor_module_latex("p", "q", manifold)
        out = _tensor_module_latex("r+p", "s+q", manifold)
        lines.append(rf"{label}: {left}\times {right}\longrightarrow {out}")
    elif kind == "contraction":
        metadata = getattr(operator, "metadata", {})
        i = metadata.get("i", "?")
        j = metadata.get("j", "?")
        out = _tensor_module_latex("r-1", "s-1", manifold)
        lines.append(rf"{label}: {input_module}\longrightarrow {out}")
        lines.append(rf"\text{{contracts geometric indices }} {i}\text{{ and }}{j}")
    elif kind == "multiplication":
        lines.append(rf"{label}: {input_module}\longrightarrow {input_module}")
        lines.append(r"\text{multiplication by a fixed field or scalar}")
    elif kind == "identity":
        lines.append(rf"{label}: {input_module}\longrightarrow {input_module}")
    elif kind == "internal_action":
        internal = _internal_module_factor_latex("a", "b")
        lines.append(
            rf"{label}{free_indices}: "
            rf"{_tensor_module_latex('r', 's', manifold, internal)}"
            rf"\longrightarrow {_tensor_module_latex(r_out, s_out, manifold, internal)}"
        )
        lines.append(r"\text{matrix action on one internal index}")
    elif kind == "commutator":
        lines.append(rf"{label}{free_indices}: {input_module}\longrightarrow {output_module}")
        lines.append(r"\text{commutator } A\circ B-B\circ A")
    elif kind == "composition":
        lines.append(rf"{label}{free_indices}: {input_module}\longrightarrow {output_module}")
        lines.append(r"\text{composition; the rightmost operator acts first}")
    elif kind == "operator_contraction":
        lines.append(rf"{label}{free_indices}: {input_module}\longrightarrow {output_module}")
        lines.append(r"\text{composition followed by contraction of free operator indices}")
    elif arity != 1:
        inputs = r"\times ".join(_tensor_module_latex(chr(ord("r")+i), chr(ord("s")+i), manifold) for i in range(arity))
        lines.append(rf"{label}{free_indices}: {inputs}\longrightarrow {output_module}")
    else:
        lines.append(rf"{label}{free_indices}: {input_module}\longrightarrow {output_module}")

    _maybe_display_latex(r"\begin{aligned}" + r"\\[0.35em]".join(lines) + r"\end{aligned}")


def Display(obj, chart=None, name=None):
    """Display supported geometric objects using the most suitable formatter."""
    from tensorium.core.chart import Chart
    from tensorium.core.manifold import Manifold, MetricManifold, OpenSet
    from tensorium.fields import ValuedTensorField
    from tensorium.fields.local_tensor_field import LocalTensorField
    from tensorium.fields.tensor_field import TensorField
    from tensorium.connections import Connection, LocalAffineConnection
    from tensorium.operators import TensorOperator

    if isinstance(obj, LocalTensorField):
        return display_local_tensor(obj, name=name)
    if isinstance(obj, ValuedTensorField) or _is_internal_tensor_field_like(obj):
        return _display_internal_tensor_field(obj, chart=chart, name=name)
    if isinstance(obj, TensorField) or _is_tensor_field_like(obj):
        return display_tensor_field(obj, chart=chart, name=name)
    if isinstance(obj, LocalAffineConnection):
        return _display_local_connection(obj, name=name)
    if isinstance(obj, Connection) or _is_connection_like(obj):
        return _display_connection(obj, name=name)
    if isinstance(obj, TensorOperator) or _is_tensor_operator_like(obj):
        return _display_tensor_operator(obj, name=name)
    if isinstance(obj, Chart):
        return _display_chart(obj)
    if isinstance(obj, OpenSet):
        return _display_open_set(obj)
    if isinstance(obj, MetricManifold):
        return _display_metric_manifold(obj)
    if isinstance(obj, Manifold):
        return _display_manifold(obj)

    try:
        _maybe_display_latex(latex(obj))
    except Exception:
        print(obj)

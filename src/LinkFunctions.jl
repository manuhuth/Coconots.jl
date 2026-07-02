"""
    relu(x)

Rectified linear unit with a small positive floor: `max(x, 1e-10)`.
"""
relu(x::Real) = max(x, 1e-10)

"""
    softplus(x)

Numerically stable softplus `log(1 + exp(x))`; always positive and smooth.
"""
softplus(x::Real) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    identity_func(x)

Identity function.
"""
identity_func(x) = x

"""
    get_link_function(link_function)

Link function for the innovation rate. Allowed names: `"log"` (returns
`exp`), `"identity"`, `"relu"` and `"softplus"`.
"""
function get_link_function(link_function::AbstractString)
    link_function == "log" && return exp
    link_function == "identity" && return identity_func
    link_function == "relu" && return relu
    link_function == "softplus" && return softplus
    error("Unknown link function: $link_function")
end

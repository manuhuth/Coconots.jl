"""
    exponential_function(x)

Applies the exponential function element-wise. Commonly used as a link function in integer autoregressive models.

# Arguments
- `x::Real or AbstractArray`: Input value(s) for which the exponential should be computed.

# Returns
- `Real or AbstractArray`: Exponential of input.nential_function(1.0) # returns exp(1.0)
```
"""
function exponential_function(x)
    return exp(x)
end

"""
    logistic_function(x)

Computes the logistic function (sigmoid) element-wise.

# Arguments
- `x::Real or AbstractArray`: Value(s) at which the logistic function should be evaluated.

# Returns
- `Real or AbstractArray`: Logistic function evaluated at input(s).
"""
function logistic_function(x)
    return 1 / (1 + exp(-x))
end

"""
    relu(x)

Computes a Rectified Linear Unit (ReLU) function, modified to avoid zero values by setting a lower bound at `1e-10`.

# Arguments
- `x::Real`: Input value.

# Returns
- `Real`: max(x, 1e-10).
"""
function relu(x::Real)
    return max(x, 1e-10)
end

"""
    identity_func(x)

Identity function that returns the input unchanged.

# Arguments
- `x`: Any input value or type.

# Returns
- `Same type as input`: The input unchanged.
"""
function identity_func(x)
    return x
end

"""
    get_link_function(link_function)

Retrieves the appropriate link function based on the provided string identifier.

# Arguments
- `link_function::String`: Name of the link function. Allowed values are:
    - `"log"`: Exponential function (`exp`).
    - `"identity"`: Identity function.
    - `"relu"`: Rectified Linear Unit (ReLU).

# Returns
- `Function`: Corresponding link function.
"""
function get_link_function(link_function::String)
    if link_function == "log"
        return exp
    elseif link_function == "identity"
        return identity_func
    elseif link_function == "relu"
        return relu
    else
        error("Unknown link function")
    end
end

""" # Quadratic RMB ansatz

The energy of the RMB is given by the following expression:

``E(x) = -λ x^2 + b x + c·h + xW·h``

And the wave function is given by:

    ``Ψ(x)=∑_{h} exp(-λ x^2 + b x + c·h + xW·h)=exp(-λ x^2 + b x)∏_{i=1}^{M} (1+exp(c[i]+xw[i]))``

### Inputs:
- x: position
- λ: quadratic bias
- b: linear bias
- c: array of hidden unit biases
- w: array of weights 

### Returns:

- Ψ(x): wave function evaluated at x


#### Note: 
- The number of hidden units is given by the length of the array c, which should be equal to the length of the array w.
- In general, λ, b, c and w are _complex_ numbers.

#### Examples:
```julia
x = 0.5 # arbitrary number
λ = 1.0
b = 0.0
c = [0.0, 0.0]
w = [0.0, 0.0]
output = Ψ_G(x, λ, b, c, w) # output will be a (not normalized) gaussian

λ = 0.0
b = -1.0
c = [0.0, 0.0]
w = [0.0, 0.0]
output = Ψ_G(x, λ, b, c, w) # output will be an (not normalized) decreasing exponential
```
"""
function Ψ_G(x, λ, b, c,w)
    M = length(c) # number of hidden units
    prod = 1
    for i in 1:M
        prod *= 1+exp(c[i]+x*w[i])
    end
    return exp(-λ*x^2+x*b)*prod
end

#----------Mathematical tools----------#

""" # Simpson's rule

Numerical integration of a function using Simpson's rule.

### Inputs:
- f: array of function values
- h: step size

"""
function Simpson(f,h)
    Nf = length(f)
    s  = 2.0*ones(Nf)
    s[2:2:end-1] .*= 2.0
    s[1] = s[end]  = 1.0
    sum(s.*f)*h/3.0
end

""" # Simulated annealing
"""

function Simulated_Annealing(f, x0, T0, N, α)
    dim = length(x0)
    x = x0
    T = T0
    chain = zeros(N,dim)
    for i in 1:N
        x_new = x .+ randn(dim)
        ΔE = f(x_new) - f(x)
        if ΔE < 0
            x = x_new
        else
            if rand() < exp(-ΔE/T)
                x = x_new
            end
        end
        T *= α

        chain[i,:] = x
    end
    return x, chain
end

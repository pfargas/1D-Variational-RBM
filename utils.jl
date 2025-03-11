using ProgressMeter
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

"""# Numerical derivative
The derivative of a function f(x) is defined as
f'(x) = lim_{h->0} (f(x+h) - f(x))/h
In practice, we can discretize space such that h = x_{i+1} - x_i
and approximate the derivative as
f'(x_i) = (f(x_{i+1}) - f(x_i))/h

This function computes the derivative of a function f(x) given a vector of values f(x) at different points x.

    ## Inputs:
    - f: vector of values of the function f(x) at different points x
    - xmin: minimum value of x
    - xmax: maximum value of x
    - n: number of points in which the derivative is computed

    ## Outputs:
    - df: vector of values of the derivative of f(x) at different points x

"""
function df(f, xmin=-10, xmax=10, n=100000)
    x = range(xmin, xmax, length=n)
    x_vec = Array(x)
    h = sum(x[2:end].-x[1:end-1])/(length(x)-1)
    return (f[2:end] .- f[1:end-1]) ./ (h)
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

function wavefunction_optimizer(λ0, b0, c0, w0, T0, N, step ,learning_rate=0.99, xmin=-10, xmax=10, n=100000)
    N_hidden = length(c0)
    λ = λ0
    λ_new = λ0
    b = b0
    c = c0
    w = w0
    T = T0
    @showprogress for i in 1:N
        #λ_new = λ + step*randn()
        b_new = b .+ step*Complex.(randn(),randn())
        c_new = c .+ step*Complex.(randn(N_hidden),randn(N_hidden))
        w_new = w .+ step*Complex.(randn(N_hidden),randn(N_hidden))
        ΔE = -(energy(λ_new, b_new, c_new, w_new,xmin, xmax, n) - energy(λ, b, c, w,xmin, xmax, n))
        if ΔE < 0.
            #λ = λ_new
            b = b_new
            c = c_new
            w = w_new
        else
            if rand() < exp(-ΔE/T)
            #    λ = λ_new
                b = b_new
                c = c_new
                w = w_new
            end
        end
        if i%100 == 0
            println("Step: ", i, " Energy: ", energy(λ, b, c, w,xmin, xmax, n))
        end
        T *= learning_rate
    end
    return λ, b, c, w
end

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
    return exp.(-λ*x^2 .+ x*b)*prod
end

""" # Physical energy
"""
function energy(λ, b, c, w, xmin=-10, xmax=10, n=100000)
    x = range(xmin, xmax, length=n)
    x_vec = Array(x)
    h = sum(x[2:end].-x[1:end-1])/(length(x)-1)
    y = Complex.(ones(length(x_vec)))
    y = Complex.(ones(length(x_vec)))
    for (i,x_i) in enumerate(x_vec)
        y[i] = Ψ_G(x_i,λ,b,c,w)
    end
    d2ψ  =  (y[3:end] .- 2.0*y[2:end-1] .+ y[1:end-2]) ./ h^2  # FIXME: how can it be defined for an imaginary wf?
    Tloc = -0.5.*d2ψ ./ y[2:end-1] # local kinetic energy
    Vloc =  0.5.*x[2:end-1].^2 # local potential energy
    Eloc =  Tloc .+ Vloc # local energy
    return Simpson(abs.(y[2:end-1]).^2 .* Eloc,h) / Simpson(abs.(y[2:end-1]).^2,h) # expectation value of the energy
end
        

    

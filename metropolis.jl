"""
# Metropolis-Hastings algorithm

Function that implements the Metropolis-Hasting algorithm for a given function f(x) and initial guess x.

Args:
    Nsteps: Number of steps to run the algorithm.
    f: Function to be sampled from.
    initial_guess: Initial guess for the minimization.
    step_size: Step size for the random walk.
    gaussian_step: If true, the random walk is done with a Gaussian distribution. If false, it's done with a uniform distribution.
    args: Additional arguments for the function f.

Returns:
    x: The final value of the minimization.
    chain: The chain of values sampled during the algorithm.
"""
function metropolis(Nsteps, f, initial_guess, step_size, gaussian_step=true ,args...)
    x = initial_guess
    chain = zeros(Nsteps)
    for i in 1:Nsteps
        if gaussian_step
            x_new = x + step_size*randn()
        else
            x_new = x + step_size*rand()
        end
        ΔE = f(x_new)/f(x)
        if ΔE > 1
            x = x_new
        else
            if rand() < ΔE
                x = x_new
            end
        end
        chain[i] = x
    end
    return x, chain
end

function metropolis_energy(Nsteps, energy, initial_guess, step_size, gaussian_step=true ,T=1.f0, args...)
    x = initial_guess
    chain = zeros(Nsteps)
    for i in 1:Nsteps
        if gaussian_step
            x_new = x + step_size*randn()
        else
            x_new = x + step_size*rand()
        end
        ΔE = energy(x_new, args...) - energy(x,args...)
        if ΔE < 0
            x = x_new
        else
            if rand() < exp(-ΔE/T)
                x = x_new
            end
        end
        chain[i] = x
    end
    return x, chain
end

if abspath(PROGRAM_FILE) == @__FILE__

    # Example : a normal distribution

    f(x) = exp(-x^2)
    initial_guess = 0.0
    Nsteps = 1000000
    step_size = 0.1
    # x, chain = metropolis(Nsteps, f, initial_guess, step_size)
    x, chain = metropolis_energy(Nsteps, f, initial_guess, step_size)
    x = range(-3, 3, length=1000)


    using PyPlot
    clf()
    # plot(x, 1/sqrt(2*π)*f.(x), label="True distribution")
    hist(chain, bins=100, density=true)
    xlabel("x")
    ylabel("Value")
    # logscale y
    # yscale("log")
    title("Metropolis-Hastings algorithm")
    savefig("out/metropolis.png")
end
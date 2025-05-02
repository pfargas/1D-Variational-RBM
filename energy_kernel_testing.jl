using ProgressMeter
using Random #, Distributions
using PyPlot
using CUDA
using StaticArrays
D = 2

function mock_energy_CUDA(x, energies)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    Nchains = size(x, 1)
    D = size(x, 2)

    #for i in index:stride:Nchains # NO ENTENC COM FUNCIONA AIXO
        if i < Nchains
            # Inlining the Gaussian landscape calculation directly into the kernel
            μ1 = zero(Float64)  # μ1 = (0,0) (zero vector)
            μ2 = 5.0  # μ2 = (5,5) (scalar for simplicity)
            
            s1 = zero(Float64)
            s2 = zero(Float64)
            
            # Compute squared differences directly in the kernel
            for j in 1:D
                idx = (i-1)*D + j
                s1 += (x[idx] - μ1)^2
                s2 += (x[idx] - μ2)^2
            end
            energy = exp(-0.5 * s1) + 0.5 * exp(-0.5 * s2)  # Combine the Gaussian terms
            energies[i] = energy
        end

   # end

    return
end

function mock_energy(x)
    Nchains = size(x, 1)
    D = size(x, 2)
    energies = zeros(Nchains)
    for i in 1:Nchains
        # Inlining the Gaussian landscape calculation directly into the kernel
        μ1 = zero(Float64)  # μ1 = (0,0) (zero vector)
        μ2 = 5.0  # μ2 = (5,5) (scalar for simplicity)
        
        s1 = zero(Float64)
        s2 = zero(Float64)
        
        # Compute squared differences directly in the kernel
        for j in 1:D
            s1 += (x[i,j] - μ1)^2
            s2 += (x[i,j] - μ2)^2
        end
        energy = exp(-0.5 * s1) + 0.5 * exp(-0.5 * s2)  # Combine the Gaussian terms
        energies[i] = energy
    end
    return energies
end

NTemps = 1000
test = zeros(NTemps)
x_tests = randn(NTemps, 2) # random initial guess
energies_cpu = mock_energy(x_tests)
x_tests = vec(x_tests)

test = CuArray(test)
x_tests = CuArray(x_tests)

threads = 256

blocks = cld(NTemps, threads)

@cuda threads=threads blocks=blocks mock_energy_CUDA(x_tests, test)

# Copy the result back to the host
energies = Array(test)

# Check if the results are the same
println("CUDA result: ", energies[1:10])
println("CPU result: ", energies_cpu[1:10])


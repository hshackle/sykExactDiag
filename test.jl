include("./SYK.jl")
using .SYK
using LinearAlgebra
using Statistics
using Polynomials
using Plots

# Run a few tests to compare to literature

# Test complex SYK Hamiltonian against arxiv:1910.14099
# GSE for N=10 should be -1.105 + 0.0489*Q^2

N = 10
Nfs = 2:8
samples = 1000
energies = zeros(Complex{Float64}, length(Nfs), samples)
for (i, Nf) in enumerate(Nfs)
    for j in 1:samples
        H = complexSYKHamiltonian(N, Nf, 1.0, 0.0, j)
        energies[i, j] = eigvals(H)[1]
    end
end
meanEnergies = dropdims(mean(energies, dims=2), dims=2)
println("Mean energies: ", meanEnergies)
# Fit to quadratic form
Qs = (Nfs .- N/2)
p = fit(Qs, real.(meanEnergies), 2)
println("Fit: ", p)
println("Should be -1.105 + 0.0489*Q^2")

# Now test Majorana SYK via arxiv:1806:10145
# Not the most rigorous test, but if you squint at their N=24 ground state energy, 
# it looks to be around -1.14

N = 24
samples = 20
energies = zeros(Complex{Float64}, samples)
for j in 1:samples
    H = majoranaSYKHamiltonian(N, 1.0, j)
    energies[j] = eigvals(H)[1]
end
meanEnergy = mean(energies)
println("Mean energy: ", meanEnergy, " should be around -1.14")


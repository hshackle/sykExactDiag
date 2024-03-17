module SYK

# Import linTable.jl
include("./linTable.jl")
using Random
using Reexport
using Plots

export majoranaSYKHamiltonian, complexSYKHamiltonian

# Generates the Hamiltonian for the Majorana SYK model and complex SYK model

function majoranaSYKHamiltonian(N::Int, J::Float64, seed::Int)
    Random.seed!(seed)
    # Initialize the lin table
    lt = majoranaLinTable(floor(Int, N/2))
    manyBodyDim = lt.manyBodyDim
    H = zeros(Complex{Float64}, manyBodyDim, manyBodyDim)

    for i in 1:N
        for j in (i+1):N
            for k in (j+1):N
                for l in (k+1):N
                    Jijkl = sqrt(6) * J / sqrt(N^3) * randn() 
                    for state in lt.allStates
                        newstate, sign = fourMajoranaOperator(state, i, j, k, l)
                        # Print old and new states in binary

                        H[getConfigIndex(lt, state), getConfigIndex(lt, newstate)] += Jijkl * sign
                    end
                end
            end
        end
    end
    return H
end

function complexSYKHamiltonian(N::Int, Nf::Int, J::Float64, t::Float64, seed::Int)
    Random.seed!(seed)
    # Initialize the lin table
    lt = complexLinTable(N, Nf)
    manyBodyDim = lt.manyBodyDim
    H = zeros(Complex{Float64}, manyBodyDim, manyBodyDim)

    # Generate random couplings from a complex Gaussian distribution
    Js = zeros(Complex{Float64}, N, N, N, N)
    ts = zeros(Complex{Float64}, N, N)
    for i in 1:N
        for j in (i+1):N
            for k in 1:N
                for l in (k+1):N
                    if (i == k) && (j == l)
                        Js[i, j, k, l] = J / (N^(3/2)) * (randn() )
                    elseif Js[k, l, i, j] == 0
                        Js[i,j,k,l] = J / (N^(3/2)) * (randn() + im*randn())
                    else
                        Js[i,j,k,l] = conj(Js[k, l, i, j])
                    end
                end
            end
        end
    end
    for i in 1:N
        for j in 1:N
            if ts[j, i] != 0
                ts[i,j] = conj(ts[j,i])
            else
                ts[i,j] = t * (randn() + im*randn())
            end
        end
    end

    # Renormalize hopping matrix by interactions terms to ensure PH symmetry
    for i in 1:N
        for j in (i+1):N
            for k in 1:N
                for l in (k+1):N
                    if i == k
                        ts[j, l] += 0.5 * Js[i, j, k, l]
                    end
                    if i == l
                        ts[j, k] -= 0.5 * Js[i, j, k, l]
                    end
                    if j == k
                        ts[i, l] -= 0.5 * Js[i, j, k, l]
                    end
                    if j == l
                        ts[i, k] += 0.5 * Js[i, j, k, l]
                    end
                end
            end
        end
    end
    # Now add in hoppings
    for i in 1:N
        for j in 1:N
            for state in lt.allStates
                if kthBitSet(state, j) && (!kthBitSet(state, i) | (i == j))
                    newstate, fermiFactor = complexQuadraticHopping(state, i, j)
                    H[getConfigIndex(lt, newstate), getConfigIndex(lt, state)] += ts[i, j]* fermiFactor
                end
            end
        end
    end
    # Now add in interactions
    for i in 1:N
        for j in (i+1):N
            for k in 1:N
                for l in (k+1):N
                    for state in lt.allStates
                        if kthBitSet(state, k) && kthBitSet(state, l) 
                            annihilationBitmask = UInt32((1 << (k-1)) ⊻ (1 << (l-1)))
                            newState = annihilationBitmask ⊻ state
                            if !kthBitSet(newState, i) && !kthBitSet(newState, j)
                                newState, fermiFactor = complexSYKOperator(state, i, j, k, l)
                                H[getConfigIndex(lt, newState), getConfigIndex(lt, state)] += fermiFactor * Js[i,j,k,l]
                            end
                        end
                    end
                end
            end
        end
    end
    return H
end
end # module 

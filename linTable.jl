using LinearAlgebra

abstract type LinTable end

# Implements a Lin table, which is just an efficient way of indexing many-body states with a conservation law.
# For complex fermions, we conserve fermion number, so, for example, N = 4, Nf = 2, we label each configuration
# with a number from 1 to 6, with:
#  1 -> 0011
#  2 -> 0101
#  3 -> 0110
#  4 -> 1001
#  5 -> 1010
#  6 -> 1100
# Lin tables let us quickly find the index of a given configuration, and vice versa.
# It's probably overkill for Majorana fermions where we only conserve parity, but I may as well use it.
# Main usage is getConfigIndex(lt, config) -> index, and lt.allStates[index] -> config

struct complexLinTable <: LinTable
    # Initializes LinTable for N sites, with Nf fermions
    N::UInt32
    Nf::UInt32
    # Useful quantities to have precomputed 
    halfN::UInt32
    manyBodyDim::UInt32
    allStates::Vector{UInt32} 
    leftHash::Vector{UInt32}
    rightHash::Vector{UInt32}
    rightMask::UInt32

    function complexLinTable(N::Int, Nf::Int)
        @assert Nf <= N
        @assert iseven(N)
        halfN = UInt32(N >> 1)
        manyBodyDim = binomial(N, Nf)
        allStates = zeros(UInt32, manyBodyDim)
        leftHash = zeros(UInt32, 2^halfN)
        rightHash = zeros(UInt32, 2^halfN)
        rightMask = UInt32(2^halfN - 1)
        nIdxes = zeros(UInt32, N)

        # First, go through right config and compute hash
        for config in 0:2^halfN-1
            rightNf = count_ones(config)
            rightHash[config+1] += nIdxes[rightNf+1]
            nIdxes[rightNf+1] += 1
        end


        idx = 1
        for config in 0:2^N-1
            if count_ones(config) == Nf
                allStates[idx] = config
                leftConfig = config >> halfN
                rightConfig = config & rightMask
                leftHash[leftConfig + 1] = idx - rightHash[rightConfig+1]
                idx += 1
            end
        end

        # Check to make sure everything works
        idx = 1
        for config in 0:2^N-1
            if count_ones(config) == Nf
                leftConfig = config >> halfN
                rightConfig = config & rightMask
                @assert leftHash[leftConfig+1] + rightHash[rightConfig+1] == idx
                idx += 1
            end
        end
        new(N, Nf, halfN,manyBodyDim, allStates, leftHash, rightHash, rightMask)
    end
end

struct majoranaLinTable <: LinTable
    # Initializes LinTable for N sites
    # We no longer have conserved fermion number, but we do have conserved parity which we assume is even
    N::Int
    # Useful quantities to have precomputed 
    halfN::Int
    manyBodyDim::Int
    allStates::Vector{UInt32} 
    leftHash::Vector{UInt32}
    rightHash::Vector{UInt32}
    rightMask::UInt32

    function majoranaLinTable(N::Int)
        @assert iseven(N)
        halfN = N >> 1
        manyBodyDim = 2^(N-1)
        allStates = zeros(UInt32, manyBodyDim)
        leftHash = zeros(UInt32, 2^halfN)
        rightHash = zeros(UInt32, 2^halfN)
        rightMask = UInt32(2^halfN - 1)
        rightParityIdxes = zeros(UInt32, 2)

        # First, go through right config and compute hash
        for config in 0:2^halfN-1
            rightParity = iseven(count_ones(config))
            rightHash[config+1] += rightParityIdxes[rightParity+1]
            rightParityIdxes[rightParity+1] += 1
        end

        idx = 1
        for config in 0:2^N-1
            if iseven(count_ones(config))
                allStates[idx] = config
                leftConfig = config >> halfN
                rightConfig = config & rightMask
                leftHash[leftConfig + 1] = idx - rightHash[rightConfig+1]
                idx += 1
            end
        end

        # Check to make sure everything works
        idx = 1
        for config in 0:2^N-1
            if iseven(count_ones(config))
                leftConfig = config >> halfN
                rightConfig = config & rightMask
                @assert leftHash[leftConfig+1] + rightHash[rightConfig+1] == idx
                idx += 1
            end
        end
        new(N, halfN, manyBodyDim, allStates, leftHash, rightHash, rightMask)
    end
end

function getConfigIndex(lt::LinTable, config::UInt32)
    leftConfig = config >> lt.halfN
    rightConfig = config & lt.rightMask
    return lt.leftHash[leftConfig+1] + lt.rightHash[rightConfig+1]
end

# Implements action of Majorana and complex fermion operators on a many-body state,
# returning the new state and the Pauli sign.

function kthBitSet(n::UInt32, k::Int)
    return (n & (1 << (k-1))) != 0
end

function majoranaOperator(state::UInt32, n::Int)
    # Applies the n-th Majorana operator to the state, returning the new state and Pauli sign 
    complexSite = ceil(Int, n/2)
    fermiFactor = 1 - 2 * (count_ones(state & (UInt32(2^(complexSite-1) - 1))) % 2)
    newState = state ⊻ UInt32(1 << (complexSite-1))
    if iseven(n)
        return newState, fermiFactor / sqrt(2)
    else
        fermiFactor *= im * (1 - 2 * kthBitSet(state, complexSite))
        return newState, fermiFactor / sqrt(2)
    end
end

function fourMajoranaOperator(state::UInt32, n::Int, m::Int, p::Int, q::Int)
    # Applies the n-th, m-th, p-th, and q-th Majorana operators to the state, returning the new state and Pauli sign 
    # Useful for SYK
    state1, fermiFactor1 = majoranaOperator(state, q)
    state2, fermiFactor2 = majoranaOperator(state1, p)
    state3, fermiFactor3 = majoranaOperator(state2, m)
    state4, fermiFactor4 = majoranaOperator(state3, n)
    return state4, fermiFactor1 * fermiFactor2 * fermiFactor3 * fermiFactor4
end

function complexCreationOperator(state::UInt32, n::Int)
    # Applies the n-th creation operator to the state, returning the new state and Pauli sign 
    if kthBitSet(state, n)
        return state, 0
    else
        newState = state ⊻ UInt32(1 << (n-1))
        fermiFactor = 1 - 2 * (count_ones(state & (UInt32(2^(n-1) - 1))) % 2)
        return newState, fermiFactor
    end
end

function complexAnnihilationOperator(state::UInt32, n::Int)
    # Applies the n-th annihilation operator to the state, returning the new state and Pauli sign 
    if !kthBitSet(state, n)
        return state, 0
    else
        newState = state ⊻ UInt32(1 << (n-1))
        fermiFactor = 1 - 2 * (count_ones(state & (UInt32(2^(n-1)- 1))) % 2)
        return newState, fermiFactor
    end
end

function complexSYKOperator(state::UInt32, n::Int, m::Int, p::Int, q::Int)
    # Applies the n-th, m-th, p-th, and q-th SYK operator to the state, returning the new state and Pauli sign 
    state1, fermiFactor1 = complexAnnihilationOperator(state, q)
    state2, fermiFactor2 = complexAnnihilationOperator(state1, p)
    state3, fermiFactor3 = complexCreationOperator(state2, m)
    state4, fermiFactor4 = complexCreationOperator(state3, n)
    return state4, fermiFactor1 * fermiFactor2 * fermiFactor3 * fermiFactor4
end

function complexQuadraticHopping(state::UInt32, n::Int, m::Int)
    # Applies the n-th and m-th quadratic hopping operator to the state, returning the new state and Pauli sign 
    state1, fermiFactor1 = complexAnnihilationOperator(state, m)
    state2, fermiFactor2 = complexCreationOperator(state1, n)
    return state2, fermiFactor1 * fermiFactor2
end

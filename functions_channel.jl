using LinearAlgebra
using QuantumInformation
using StatsBase

include("functions.jl")

function measure_comp(state)
    """
    measures the state given in the computational basis and returns the index of outcome and the density matrix of the outcome, e.g. for a two level system, if 
    the |0> state was found, then it would return |0><0| == [1 0; 0 0] and the index 1, while if |1> was found, it would return |1><1| == [0 0; 0 1] and index 2
    
    Parameters
    ---
    state: the density matrix of a state, any dimension, must be square

    Returns
    ---
    i: the index refering to the basis that was found after measuring
    basis[i]: the complex matrix, the density matrix of the state that was measured

    """
    dim=size(state)[1]
    basis = Vector{Matrix{ComplexF64}}(undef, 0)
    for i in 1:dim
        append!(basis, [proj(ket(i,dim))])
    end
    M = POVMMeasurement(basis)
    probs = aweights(diag(M(state)))
    i = sample(collect(1:dim),probs)
    return i, basis[i]
end

function one_simplified_round(choi,dim)
    """
    For testing our understanding of shadow tomography on quantum channels, this performs one round of a very simplified channel, 
    where two random unitary matrices are generated, U and W, to probe the channel given by choi. 

    Parameters
    ---
    dim: dimensions of the input space, assumed to be the same as output space
    choi: the choi state of the map being probed by tomography

    Returns
    ---
    snapshot: given by (U^*|0><0|U^T) ⊗ (W^dag |b><b| W), then need to apply an inverse map to get the shadow of the map represented by choi
    """
    rho = proj(ket(1,dim))
    U = haar_measure(dim)
    W = haar_measure(dim)
    rho_end = reshape(vec_unitary(W)*vec_choi(choi, dim, dim)*vec_unitary(U)*vec(rho),(dim,dim))
    b,state_b = measure_comp(rho_end) #state_b should be a dim by dim vector
    
    first = conj(U)*rho*transpose(U)
    second = W' * state_b * W
    return convert(Matrix{Float64}, first ⊗ second) #returns (U^*|0><0|U^T) ⊗ (W^dag |b><b| W)
end

function inverse_M(mu)
    """
    MEANT ONLY AS A CHECK 
    Caclulates the inverse of the measurement channel, defined as the expectation value of the classical snapshot of a very simple channel of simply a unitary going from an input space to an 
    output space of the same dimensions. so M(rho)= E(U^dag |b><b|U) and the inverse of the measurement channel is M^(-1)(mu)= (d+1)mu - tr(mu)*I

    Parameters
    ---
    mu: a square matrix of dimensions d, something like U^dag |b><b|U

    Returns
    ---
    inverse: given by (d+1)*(mu - I*tr(mu))

    """
    d = size(mu)[1]
    Id =  Matrix{Int64}(I,d,d)
    return((d+1)*mu) - (Id*tr(mu))
end

function inverse_M_vec(d)
    """
    The inverse map of the measurement channel as a matrix of dimensions d^2 to be applied to a vectorised matrix of dimensions d, defined as
    M^(-1)_v*vec(mu) = vec(M^(-1)(mu)). THe measurement channel is defined as the expectation value of the classical snapshot of the channel 
    of a unitary, so M(rho) M(rho)= E(U^dag |b><b|U) and the inverse of the measurement channel is M^(-1)(mu)= (d+1)mu - tr(mu)*I

    Parameters
    ---
    dim: the dimensions of the space the map translates to and from 
    Returns
    ---
    inverse: a matrix of dimensions d^2, given the vectorised version of M^(-1)(mu) = (d+1)*(mu - I*tr(mu))
    """
    I_v = Matrix{Int64}(I,d^2,d^2) #vectorised identity matrix 
    I_d = vec(Matrix{Int64}(I,d,d))

    return ((d+1)*I_v) - (I_d * transpose(I_d))
end


function apply_inverse_M(mu,rho)
    """
    USE IF YOU CAN GET SEPARATE MU AND RHO
    Applies the inverse measurement channel to mu and to rho, 
    M^(-1)(mu)⊗M^(-1)(rho) == (M_v^(-1)⊗M_v^(-1)) * (vec(mu) ⊗ vec(rho))
    
    Parameters
    ---
    mu: a square matrix of dimensions d_1
    rho: a square matric of dimensions d_2

    Returns
    ---
    inverse: a matrix of dimesnions d_1^2 by d_2^2, given by (M_v^(-1)⊗M_v^(-1)) * (vec(mu) ⊗ vec(rho))
    """
    d_1 = size(mu)[1]
    d_2 = size(rho)[1]
    separated = reshape((inverse_M_vec(d_1) ⊗ inverse_M_vec(d_2))*(vec(mu) ⊗ vec(rho)), (d_1, d_1, d_2, d_2) )
    permuted = permutedims(separated, (1,3,2,4))
    return reshape(permuted, (d_1^2,d_2^2))
end

function apply_inverse_M_tensored(state)
    """
    USE IF YOU CAN ONLY GET THE TENSOR PRODUCT RHO ⊗ MU 
    applies the inverse measurement channel to state = mu ⊗ rho

    Parameters
    ---
    state: a square matrix of dimensions d^2, state = mu ⊗ rho

    Returns
    ---
    inverse: given by (M^(-1)(mu)⊗M^(-1)(rho)) = (M_v^(-1)⊗M_v^(-1))(vec(mu ⊗ rho)) with permutations to make these equal
    """
    d=convert(Int64,(size(state)[1]^(1/2)))
    vec_mu_rho=vec(permutedims(reshape(vec(state),(d,d,d,d)),(1,3,2,4)))
    separated = reshape((inverse_M_vec(d) ⊗ inverse_M_vec(d))*(vec_mu_rho), (d, d, d, d) )
    permuted = permutedims(separated, (1,3,2,4))
    return reshape(permuted, (d^2,d^2))
end


function channel_tomography(lambda, d ,n_rounds)
    """
    (we could probs make this more general)
    Performs shadow tomography on the channel lambda by forming random unitaries and appling them to lambda. 
    Lambda should go from input space of dimensions d to output space of same dimensions. We apply random unitaries and record
    the guess (U^*|0><0|U^T)⊗(W^dag |b><b| W) for a number n_rounds, forms the average value then performs the inverse 
    measurement channel, M^(-1) ⊗ M^(-1), onto this state. This is the shadow of the channel lambda.

    Parameters
    ---
    lambda: a choi matrix  for a map that goes from a system of dimensions d to an output system of same dimensions. 
    so a square matrix of dimensions d^2
    d: the dimensions of the input and output state
    n_rounds: the number of rounds to perform shadow tomography

    Returns
    ---
    shadow_exp: the expectation value of the shadow of gamma, given by the inverse maps M^(-1)⊗M^(-1), applied to the average of the outcome of n_rounds of 
    one_simplified_round, which gives (U*|0><0|U^T)⊗(W^+|b><b|W). 
    """
    ave = zeros(d^2,d^2)
    for i=1:n_rounds
        ave+= one_simplified_round( lambda, d )
    end
    return(apply_inverse_M_tensored(ave/n_rounds))
end

#d = 3
#rho = rand_rho(d)
#mu = rand_rho(d)

#shows how to get to vec(rho) ⊗ vec(mu) from vec(rho ⊗ mu) by permuting dimensions
#println(vec(permutedims(reshape(vec(rho ⊗ mu),(d,d,d,d)),(1,3,2,4))))
#println(vec(rho) ⊗ vec(mu))

#showing that the function apply_inverse_M(rho,mu) == M_v^(-1) ⊗ M_v^(-1) * (vec(rho) ⊗ vec(mu)) 
#println(apply_inverse_M(mu,rho))
#println(inverse_M(mu) ⊗ inverse_M(rho))

#showing how apply_inverse_M_tensored works
#apply_inverse_M_tensored(rho ⊗ mu)
#apply_inverse_M(rho,mu)

d=2
choi = rand_CPTP(d,d)
choi = Matrix{Int64}(I,d^2,d^2)
println(choi)

guess = channel_tomography(choi, d , 10000000)
#these are not equal ...
isapprox.(choi,guess,rtol=0.2 )

println(guess)
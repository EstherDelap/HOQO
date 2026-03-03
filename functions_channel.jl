using LinearAlgebra
using QuantumInformation
using StatsBase
using Plots

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
    for ind in 1:dim
        append!(basis, [proj(ket(ind,dim))])
    end
    M = POVMMeasurement(basis)
    probs = aweights(diag(M(state)))
    i = sample(collect(1:dim),probs)
    return i, basis[i]
end

function one_simplified_round(choi,dim; state = false)
    """
    Performs one round of shadow tomography on channel or state represented by choi,
    by generating random unitaries U and W 

    Parameters
    ---
    dim: dimensions of the input space, assumed to be the same as output space
    choi: the choi state of the map being probed by tomography
    state: (optional) bool, either choi is a state or a channel
    separated (optional) 

    Returns
    ---
    snapshot: given by (U^*|0><0|U^T) ⊗ (W^dag |b><b| W)
    """
    W = haar_measure(dim) #random unitary matrix sampled over the haar measure 
    if state == false #for shadow tomography on a channel 
        U = haar_measure(dim)
        zero = proj(ket(1,dim))
        rho_end = reshape(vec_unitary(W)*vec_choi(choi, dim, dim)*vec_unitary(transpose(U))*vec(zero),(dim,dim))
        first = conj(U)*zero*transpose(U)
    else #for shadow tomography on a state
        rho_end = reshape(vec_unitary(W)*vec(choi),(dim,dim))
        first = [1]
    end
    b,state_b = measure_comp(rho_end) #state_b should be a dim by dim vector
    second = W' * state_b * W
    return first⊗second
end

function inverse_M_vec(d)
    """
    The inverse map of the measurement channel as a matrix of dimensions d^2 to be applied to a vectorised matrix of dimensions d, defined as
    M^(-1)_v*vec(mu) = vec(M^(-1)(mu)).
    Parameters
    ---
    d: the dimensions of the space the map translates to and from 
    Returns
    ---
    inverse: a matrix of dimensions d^2, given the vectorised version of M^(-1)(mu) = (d+1)*(mu - I*tr(mu))
    """
    I_v = Matrix{Int64}(I,d^2,d^2) #vectorised identity matrix 
    I_d = vec(Matrix{Int64}(I,d,d))

    return ((d+1)*I_v) - (I_d * transpose(I_d))
end


function inverse_M_1_4(state)
    """
    applies the inverse measurement channel M^(-1)⊗ M^(-4) to state = mu ⊗ rho

    Parameters
    ---
    state: a square matrix of dimensions d^2, state = mu ⊗ rho

    Returns
    ---
    inverse: given by (M^(-1)(mu)⊗M^(-1)(rho)) = (M_v^(-1)⊗M_v^(-1))(vec(mu ⊗ rho)) with permutations to make these equal
    """
    
    d = convert(Int64,(size(state)[1]^(1/2)))
    vec_mu_rho = vec(permutedims(reshape(vec(state),(d,d,d,d)),(1,3,2,4)))
    separated = reshape( ( (d*inverse_M_vec(d)) ⊗ (inverse_M_vec(d)) ) * (vec_mu_rho), (d, d, d, d) )
    permuted = permutedims(separated, (1,3,2,4))
    return reshape(permuted, (d^2,d^2))
end

function M_vec(d)
    """
    Vectorised version of the map M^(4), so M^(4)[rho] = M^(4)_vec * |rho>>
    """
    I_v = Matrix{Int64}(I,d^2,d^2) #vectorised identity matrix 
    I_d = vec(Matrix{Int64}(I,d,d)) #for computing the trace

    return (1/(d+1))*(I_v + (I_d * transpose(I_d)))
end

function M_1_4(gamma,d)
    """
    forms a vectorised version of the two maps M^(1)⊗M^(4)[choi]  
    """
    vec_mu_rho = vec(permutedims(reshape(vec(gamma),(d,d,d,d)),(1,3,2,4)))
    separated = reshape( ( map_M_vec(d) ⊗ (map_M_vec(d)/d) )*(vec_mu_rho) , (d, d, d, d) )
    permuted = permutedims(separated, (1,3,2,4))
    return reshape(permuted, (d^2,d^2))
end


function channel_tomography(lambda, d ,n_rounds, existing_ave_inversed, existing_ave_mapped; is_state = false)
    """
    Performs shadow tomography on the channel or state lambda by forming random unitaries and appling them to lambda. 
    Record the guess (U^*|0><0|U^T)⊗(W^dag |b><b| W) or (W^dag |b><b| W) for a number n_rounds, form the average value 
    then perform the inverse measurement channel, M^(-1) ⊗ M^(-4) or M^(-4). 

    Parameters
    ---
    lambda: a choi matrix of the map, or the density matrix of the state
    d: the dimensions of the input and output state
    n_rounds: the number of rounds to perform shadow tomography
    existing_ave_inversed: the existing average of M^(-1,-4)[(U^*|0><0|U^T)⊗(W^dag |b><b| W)]
    existing_ave_mapped: the exisitng average shadow (U^*|0><0|U^T)⊗(W^dag |b><b| W) 

    Returns
    ---
    ave: the average of (U*|0><0|U^T)⊗(W^+|b><b|W) or (W^+|b><b|W)
    shadow: the inverse map  M^(-1)⊗M^(-4) or M^(-4) applied to the ave
    """
    if is_state == false #for shadow tomography on a channel
        ave = zeros(d^2,d^2)
    else #for shadow tomography on a state
        ave = zeros(d,d)
    end

    for i=1:n_rounds
        ave+= one_simplified_round( lambda, d, state = is_state)
    end

    ave=ave/n_rounds
    
    if is_state == false
        return( (inverse_M_1_4(ave) + existing_ave_inversed)/2, (ave+existing_ave_mapped)/2 )
    else
        return(reshape(inverse_M_vec(d)*vec(ave), (d,d)))
    end
end

function plot_shots(choi, d, total_n; mu = I, start = 1, state= false)
    """
    plots the number of shots agianst the relative error, where the number of shots increases as 10^n with n in total_n.

    Parameters
    ---
    choi: the choi matrix of the channel we are performing shadow tomography on
    total_n: the number of rounds perfromed, where one round will perform shadow tomography for 10^n rounds
    Returns
    ---
    shots: a list of the numebr of shots, so shots=(10^1,10^2,10^3, ..., 10^total_n)
    rel_err: the relative error for each shadow tomography attempt, so a list where each item is of the form 
    norm(choi-shadow)/norm(shadow), where shadow = channel_tomography for 10^n rounds 
    """
    shots = zeros(1) #x axis
    rel_err_inv_map = zeros(0) #y axis comparing choi with M^(-1,-4)(average)
    rel_err_map = zeros(0) #second y axis comparing M^(1,4)(choi) with average
    
    if state == false #for shadow tomography on a channel
        ave_inverse = zeros(ComplexF64,d^2,d^2)
        ave_mapped = zeros(ComplexF64,d^2,d^2)
        choi_mapped = M_1_4(choi,d)
    else #for shadow tomography on a state
        ave_inverse = zeros(d,d)
        ave_mapped = zeros(d,d)
        choi_mapped = reshape(M_vec(d)* vec(choi), (d,d))
    end

    for n=start:total_n
        num_rounds = 10^n
        append!(shots,num_rounds )

        ave_inverse, ave_mapped = channel_tomography(choi, d, num_rounds-shots[end-1], ave_inverse, ave_mapped, is_state=state)
            
        append!(rel_err_inv_map, norm( choi-ave_inverse) / norm(choi))
        append!(rel_err_map, norm( choi_mapped-ave_mapped) / norm(choi_mapped) )
    end
    return deleteat!(shots,1), rel_err_inv_map, rel_err_map
end

total_num_shots = 7

dim = 2
rho = rand_rho(dim)
choi = rand_CPTP(dim,dim)

x, y, y_2= plot_shots(choi, dim, total_num_shots, start = 3)

plot(x, [y y_2], xaxis = :log10, layout=(2, 1), legend=false)
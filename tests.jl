include("functions.jl")
include("functions_channel.jl")
using StatsPlots
using RandomMatrices

function act_map_choi(choi, rho, dim_in, dim_out)
    """
    Given the choi matrix of a quantum map going from space with dimensions dim_in to space
    with dimensions dim_out, the action of the map is computed on the state rho.

    Parameters
    ---
    choi: a dim_in*dim_out dimensional matrix representing the choi state of a quantum map
    rho: a density matrix of dimensions dim_in, our quantum state
    dim_in: index, the dimensions of the input space of the quantum map
    dim_out: index, the dimensions of the output space of the quantum map

    Returns
    ---
    rho_new: the outcome of the quantum map acting on rho, given by tr_in(choi(I_out otimes rho^T)) 
    """
    total = (Transpose(rho) ⊗ Matrix{Int64}(I,dim_out,dim_out) ) * choi 

    return ptrace(total, [dim_in, dim_out], 1)
end


function testing_probabilities(rho, dim_aux, dim_env, dim_sys )
    """
    test the function prob_b by using rho' (i.e. rho once the unitary has been applied to the system and auxilliary space) then takes the partial trace
    over the system and environement, and finally projects this onto the computational basis

    Parameters
    ---
    rho: a square matrix of dimensions dim_aux*dim_sys*dim_env, the tripartite state once the unitary has been applied to the system and auxiliary space
    
    Returns
    ---
    probabilities: a list of dimensions dim_aux that should be the probabilities 

    """
    rho_aux = ptrace( rho, [dim_env,dim_sys,dim_aux],[1,2])
    probs = zeros(0)
    for i in range(1,dim_aux)
        basis = zeros(dim_aux); basis[i] = 1
        prob_i = basis' * rho_aux * basis
        append!(probs, prob_i)
    end
    return probs
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


function map_M(rho,d)
    """

    """
    Id = Matrix{Int64}(I,d,d)
    return (1/(d+1))*((tr(rho)*Id)+rho)
end

function eigval_dist(num_rounds, dim)
    eigs = zeros(0)
    for n=1:num_rounds
        eigs = cat( eigs, eigvals(haar_measure(dim)),dims=1)
        #eigs = cat( eigs, eigvals(rand(CUE(dim))),dims=1)
        #eigs = cat(eigs, eigvals(rand( Haar(2), dim, 1) ),dims=1) 
    end
    return eigs
end


#CODE TO CHECK VEC_UNITARY WORKS
#rho_dashed_normal = vec(U*rho*U')
#rho_dashed_vec = vec_unitary(U) * vec(rho) #this is a vectorised version of rho' = U rho U^dag
#println(norm(rho_dashed_normal-rho_dashed_vec))

#CODE TO CHECK VEC_CHOI WORKS
#dim = 4
#choi = rand_CPTP(dim,dim)
#rho = rand_rho(dim)
#zero = proj(ket(1,dim))
#Id = Matrix{Int64}(I,dim,dim) #I_out 
#rho_dashed_1 = vec(ptrace( (transpose(rho) ⊗ Id) * choi , [dim,dim], [1]) )
#rho_dashed_2 = vec_choi(choi,dim,dim)*vec(rho)
#println(norm(rho_dashed_1-rho_dashed_2 ))

#CODE TO CHECK VEC_UNITARY(w)*VEC_CHOI(CHOI)*VEC_UNITARY(U^DAG)*VEC(ZERO) WORKS 
#dim = 4
#rho = rand_rho(dim)
#zero = proj(ket(1,dim))
#U = haar_measure(dim)
#W = haar_measure(dim)
#choi = rand_CPTP(dim,dim)
#Id = Matrix{Int64}(I,dim,dim) 
#rho_3 = W*ptrace( (transpose(transpose(U)*zero*conj(U)) ⊗ Id)*choi, [dim,dim],[1])*W'
#rho_1 = W*ptrace(choi*(Id  ⊗ transpose(transpose(U)*zero*conj(U))),[dim,dim],[2])*W'
#rho_2 = reshape(vec_unitary(W)*vec_choi(choi,dim,dim)*vec_unitary(transpose(U))*vec(zero),(dim,dim))
#println(norm(rho_3-reshape(rho_2,(dim,dim))))

#SHOWS HOW TO GET VEC(RHO)  ⊗ VEC(MU) FROM VEC(RHO  ⊗  MU) BY PERMUTING DIMENSIONS
#d=4
#rho = rand_rho(d)
#mu = rand_rho(d)
#murho1 = vec(permutedims(reshape(vec(rho ⊗ mu),(d,d,d,d)),(1,3,2,4)))
#murho2 = vec(rho) ⊗ vec(mu)
#println(norm(murho1-murho2))

#SHOWING INVERSE_M WORKS
#d=4
#mu=haar_measure(d)
#rho=haar_measure(d)
#mu_1 = vec(inverse_M(mu))
#mu_2 = inverse_M_vec(d) * vec(mu)
#println(norm(mu_1-mu_2))
#mu_3 = vec(inverse_M_1(mu))
#mu_4 = d*inverse_M_vec(d) * vec(mu)
#println(norm(mu_3-mu_4)/norm(mu_3))

#SHOWING HOW APPLY_INVERSE_M_TENSORED WORKS
#M_1 = apply_inverse_M_tensored(mu ⊗ rho)
#M_2 = (d*inverse_M(mu)) ⊗ inverse_M(rho)
#println(norm(M_2-M_1))

#TO CHECK THE HAAR MEASURE WORKS
#a = eigval_dist(10000,50)
#density(angle.(a))
#println(norm(choi - apply_inverse_M_tensored(average_shadow)) / norm(apply_inverse_M_tensored(average_shadow))) 
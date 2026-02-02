using LinearAlgebra
using QuantumInformation
using StatsBase


function rand_CPTP(dim_in, dim_out)
    """
    Produces a the Choi matrix C of a random CPTP map that goes from input space of dimensions dim_in
    to an output space of dimensions dim_out, so C is unitary (adjoint(C)*C = identity on tensor product
    of input and output space) and tracing out the output space gives an identity on the input space. 

    Parameters
    ---
    dim_in: an integer, the dimension of the iput space
    dim_out: an integer, the dimension of the output space

    Returns
    ---
    C: the choi matrix of random CPTP map going from input to output space, so C is a dim_in*dim_out 
    dimensional matrix.
    """
    D = dim_in * dim_out
    #A = randn(complex(Float64),(D,D))
    A = randn(D,D)
    #to produce a positive matrix, multiply the random matrix by its adjoint
    pos_A = adjoint(A) * A

    #trace out the output space, so sys is space 2, and dim is the dimensions of the two subspaes A is defined on
    E = ptrace(pos_A, [dim_in,dim_out],2) #defined on input space

    B = inv(sqrt(E)) ⊗ Matrix{Int64}(I,dim_out,dim_out) #defined on input and output space, ordered as (input,ouput)
    
    C = B * pos_A * B #ordered as (input, output, input, output) 

    return C
end




function rand_rho(dim)
    """
    returns the density matrix of a random pure state of dimensions dim

    Parameters
    ---
    dim: integer, the dimension of the space of the input space

    Returns
    ---
    rho: a density matrix, so a matrix of size dim by dim that obeys tr(rho^2) = 1

    """

    v = randn(dim)
    vec_norm = (1/ norm(v))*v #normalising the random vector of dimensions dim
    return(vec_norm * vec_norm') #returns the density matrix of this normalised vector
end


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


function vec_choi(choi, dim_in, dim_out)
    """
    Generates the 'vecotised' version of the choi matrix, choi_v,  defined by the equation 
    vec(choi*rho)= choi_v vec(rho)

    Parameters
    ---
    choi: the choi matrix of a map going from input space to output space, so a matrix of 
    dimensions dim_in*dim_out by dim_in*dim_out
    dim_in: the dimensions of the input space
    dim_out: the dimensions of the output space

    Returns
    ---
    choi_v: the 'vectorised' version of the choi state, so a matrix of dimensions dim_out^2 by dim_in^2
    """

    #turn the choi matrix into a tensor so we can permute it's indices
    choi_tensor = reshape(choi, (dim_out, dim_in, dim_out, dim_in)) #the choi matrix is stored as C_(i' i; j' j)

    choi_v = permutedims(choi_tensor, (1,3,2,4)) #permutes the indices to C_(i' j'; i j)

    return reshape(choi_v,(dim_out^2,dim_in^2))

end


function haar_measure(dim)
    """
    Generates a random unitary matrix of dimensions dim sampled over the Haar measure

    Parameters
    ---
    dim: an integer, the dimensions of the desired matrix

    Returns
    ---
    q: a random matrix sampled over the Haar measure, code taken from arXiv.0609050, p. 11

    """
    z = randn(Complex{Float64},(dim,dim))/sqrt(2) #random complex matrix 

    F=qr(z)
    q = F.Q #this is a unitary matrix, i.e. q*q'=idenitity
    r=F.R
    
    d = diag(r) #this is a vecotr containg the diagonal elements of r
    ph = d ./ broadcast(abs, d) #forming th diagonal matrix with each diagonal element being r_{ii}/|r_{ii}|
    
    Q = q .* ph  #element-wise matrix multiplication
    return Q
end


function vec_unitary(U)
    """
    'Vectorise' the unitary U, such that rho'=U rho U^dag == U_v vec(rho) 

    Parameters
    ---
    U: a unitary matrix, so a square matrix which obeys U*U'=i

    Returns
    ---
    U_v: a vectorised version of the unitary matrix, so a matrix of size dim^2 by dim^2
    """
    dim = size(U,1)

    U_dims = reshape(U ⊗ U', (dim,dim,dim,dim)) #break up U ⊗ U' so we can permute the dimensions

    U_v = permutedims(U_dims, (2,3,1,4))
    return reshape(U_v, (dim^2, dim^2))
end

function apply_partial_unitary_vec(rho, a, U, dim_env)
    """
    For the system-environment state rho and the auxilliary state a, applies the unitary to the system and auxilliary space using vectorisation

    Parameters
    ---
    rho: the joint system-environment state, a square matrix of dimensions dim_sys*dim_env that is positive and of unit trace.
    a: the auxilliary system initial state, a square matrix of dimensions dim_a, positive and unit trace (usually 2-dimensional)
    U: the unitary matrix applied to the system and auxilliary space, so a square matrix of dimensions dim_sys*dim_a that obeys U*U'=I

    Returns
    ---
    rho_new: a vectorised tri-partite state so a vector of size (dim_env*dim_sys*dim_a)^2, given by I_{env} ⊗ U_{sys,a} (rho_{env,sys} ⊗ a_{a}) 
    """
    state = vec(rho ⊗ a) #vectorised initial tripartite state, ordered (env,sys,aux)
    transform= vec_unitary(Matrix{Int64}( I, dim_env, dim_env) ⊗ U) #'vectorised' version of I_env ⊗ U_{sys,aux}
    return transform*state #vectorised rho' of dimensions (dim_sys*dim_env*dim_aux)^2
end    

function apply_partial_unitary(rho, a, U, dim_env)
    """
    For the system environement state rho and the auxilliary space a, applies the unitary U to the systme and auxilliary space by the usual 
    way

    Parameters
    ---
    rho: the joint system-environment state, a square matrix of dimensions dim_sys*dim_env that is positive and of unit trace.
    a: the auxilliary system initial state, a square matrix of dimensions dim_a, positive and unit trace (usually 2-dimensional)
    U: the unitary matrix applied to the system and auxilliary space, so a square matrix of dimensions dim_sys*dim_a that obeys U*U'=I

    Returns
    ---
    rho_new: a tri-partite state, so a square matrix of dimensions dim_env*dim_sys*dim_a, obtained by (I_env ⊗ U) (rho ⊗ a) (I_env ⊗ U)'
    """
    #dim_env = convert(Int,(size(rho)[1] * size(a)[1]) / size(U)[1])
    state = rho ⊗ a #dim = dim_env*dim_sys*dim_aux
    mult = Matrix{Int64}(I, dim_env, dim_env) ⊗ U #I_env ⊗ U, dim=dim_env*dim_sys*dim_aux
    return mult * state * mult' #(I_env ⊗ U) (rho ⊗ a) (I_env ⊗ U)', a square matrix of dimensions dim_env*dim_sys*dim_aux

end

function prob_b(rho, dim_sys, dim_env, dim_aux)
    """
    Calculates the probabilities of measuring the auxilliary system in states in b_array, given rho is a tri-partitie state ordered as {env,sys,aux} 
    and b_array is an array containg the values that the (two-dimensional) auxilliary system can take 

    Parameters
    ---
    rho: a tripartite state, so a 6 dimensional tensor {dim_aux, dim_sys, dim_env, dim_aux, dim_sys, dim_env}
    dim_sys: integer, the dimensions of the system
    dim_env: integer, the dimensions of the environment
    dim_aux: integer, the dimensions of the auxilliary system

    Returns
    ---
    probs: an array of length b containg the probailities of measuring the auxilliary system in the basis states of the auxiliary system, so 
    probs[1]=tr(rho[:,:,1,:,:,1]) and probs[2]=tr(rho[:,:,2,:,:,2]) (and usually the auxilliary system is two-dimensional)

    """
    # Not sute if this is correct, but the probabilities it returns do sum to 1
    # I think the spaces are ordered as {env, sys, aux} and the way julia indexes things is opposite to how we think, 
    #i therefore break it up the other way around, as {dim_aux,dim_sys,dim_env,dim_aux,dim_sys,dim_env}

    probs=zeros(0) #an empty array

    for b in range(1,dim_aux)
        rho_b=reshape(rho[b,:,:,b,:,:], (dim_sys*dim_env, dim_sys*dim_env)) #the subatrix rho_{i,i',b'';j,j',b''} then reshaped as a matrix
        append!(probs, tr(rho_b)) #tr(rho_b) should be the probability that b is meaured 
    end

    return(probs) #an array of probabilities, should (and does) sum to 1
end   

function weighted_rho(rho, probs ,dim_aux, dim_sys, dim_env)
    """
    given a tripartite rho, and the probabilities of measuring the different outcome of the auxiliary space are given by weights, samples rho according 
    to weights, returning a normalised two-partite state rho_final 

    Parameters
    ---
    rho: a tripartite state, so a rank-six tensor ordered as {dim_aux, dim_sys, dim_env, dim_aux, dim_sys, dim_env}
    probs: an array of the probabilities of measuring the different outcomes of the auxilliary system, so of length dim_aux

    Returns
    ---
    rho_new: a normalised two-partite state as a square matrix of dimensions dim_sys*dim_env

    """

    w = aweights(probs) #turned the array into type AbstractWeights (not sure if we should be using aweights or just weights)
    index_list = collect(1:dim_aux) #a vector of integers from 1 to the length of weights== dim_aux (usually 2)
    i = sample(index_list, w) #samples a value from index_list, weighted by the values in w 
    rho_final = reshape(rho[i,:,:,i,:,:], (dim_sys*dim_env, dim_sys*dim_env))
    
    return(rho_final/tr(rho_final))
end

dim_sys = 2
dim_env = 3
dim_aux = 2
rho = rand_rho(dim_sys*dim_env) #dim = dim_sys*dim_env (matrix)
a=[1,0]*[1,0]' 
U = haar_measure(dim_sys*dim_aux) #dim_sys*dim_a



#CODE TO CHECK VEC_UNITARY WORKS

#rho_dashed_normal = vec(U*rho*U')
#rho_dashed_vec = vec_unitary(U) * vec(rho) #this is a vectorised version of rho' = U rho U^dag
#println(isapprox(rho_dashed_normal, rho_dashed_vec))



#CODE TO CHECK APPLY_PARTIAL_UNITARY_VEC WORKS

rho_1 = apply_partial_unitary_vec(rho, a, U, dim_env) #takes in rho as a matrix on spaces {env, sys}, outputs a vector rho' on spaces {env, sys, aux} 
#rho_2 = vec(apply_partial_unitary(rho, a, U, dim_env))
#println(isapprox( rho_1, rho_2))

rho_1_tensor = reshape(rho_1,(dim_aux,dim_sys,dim_env,dim_aux,dim_sys,dim_env))



#CODE TO CHECK PROB_B WORKS (at least it returns probabilities that sum to 1)

probs = prob_b( rho_1_tensor , dim_sys, dim_env, dim_aux) #rho must be a rank 6 tensor 
#println(sum(probs))


#CODE TO CHECK WEIGHTED_RHO WORKS (i have no clue tbh)

rho_final = weighted_rho(rho_1_tensor, probs,dim_aux, dim_sys, dim_env) #produces a square matrix of size dim_env*dim_sys, it is a state with unit trace
#println(tr(rho_final))


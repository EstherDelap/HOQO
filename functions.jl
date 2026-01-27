using LinearAlgebra
using QuantumInformation

function rand_state(dim)
    """
    Produces a random matrix rho that represents the density matrix of a state, so rho obeys
    positivity, rho >=0 (so all its eigenvalues are positive) and is unit trace,tr(rho)=1. 

    Parameters
    ---
    dim: the dimensions of the space the state is defined on, so rho will be a dim by dim matrix

    Returns
    ---
    rho: a dim by dim random matrix that is positive and has unit trace
    """

   A = randn(complex(Float64),(dim,dim))
   pos_A = adjoint(A) * A
   return pos_A/tr(pos_A)
end

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

dim_in = 3
dim_out = 4

rho = rand_rho(dim_in)
choi = rand_CPTP(dim_in, dim_out)


rho_1 = reshape(vec_choi(choi, dim_in, dim_out) * vec(rho), (dim_out, dim_out))

rho_2 = act_map_choi(choi, rho, dim_in, dim_out)

println(rho_1)
println(rho_2)

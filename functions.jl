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
    E = ptrace(pos_A, [dim_in,dim_out],2)

    B = inv(sqrt(E)) ⊗ Matrix{Int64}(I,dim_out,dim_out)

    C = B * pos_A * B

    return C
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
    total = choi * (Matrix{Int64}(I,dim_out,dim_out) ⊗ Transpose(rho))
    return ptrace(total, [dim_in, dim_out], 1)
end

dim_in = 2
dim_out = 3
choi = rand_CPTP(dim_in, dim_out)
rho = [ 1 1; 2 1]
act_map_choi(choi, rho, dim_in, dim_out)


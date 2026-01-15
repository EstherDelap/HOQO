using LinearAlgebra
using Tensorial

function trace_x(rho,sys, dim)
    """
    NOT FINISHED YET
    Finds the partial trace of rho with respect to the systems in sys. Dim is a list of the dimensions of the systems rho is defined on.
    So if rho was defined on A and B, and you wanted the partial trace with respect to system B, then sys would be [2] and dim would 
    have two items in it (so length(dim) would be 2). But say if rho was defined on four spaces, then you could find the partial trace with
    respect to the second and fourth system, so sys would be [2,4] and dim would have four items in it. Say the product of the elements in 
    dim is D, then rho is a D by D matrix.
    
    Parameters
    ---
    rho: the density matrix of the system we are taking the partial trace of
    sys: An integer or array of integers specifying which systems we are taking the partial trace over
    dim: the dimensions of the systems that rho is defined on

    Returns
    ----
    Partial trace of rho with respect to the systems given in sys.
    """
    #Dimensions of the traced out and the remaining systems
    D = prod(dim)
    Dtrace = prod(dim[sys])
    Dremain = convert(Int, D/Dtrace)

   shfinal = Tuple([Dremain, Dtrace, Dremain, Dtrace])
    #parameters required to decompose rho into its subsystems
   le = length(dim)

   arshape = Tuple([dim;dim])

    #permutation to permute all spaces that need to be traced out to the right, 
    #so that they can be traced out in one go.
   perm = append!(deleteat!(collect(1:le),sys),sys)
   perm = append!(perm, perm.+le)

    #reshape rho, permute all spaces that need to be traced out to the right, 
    #reshape into the form [Dremain, Dtrace, Dremain, Dtrace] and trace out over
    #the Dtrace parts.
    A = reshape(rho, arshape)
    B= permutedims(A,perm)
    C = reshape(B, shfinal)
    #now need to take the trace over axis 2 and axis 4, which in python is tr(C, axis1 = 1, axis2 = 3)
end

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
    A = randn(complex(Float64),(D,D))
    pos_A = adjoint(A) * A
    #trace out the output space, so sys is space 2, and dim is the dimensions of the two subspaes A is defined on
    E = tr_x(pos_A, 2, [dim_in, dim_out]) #this wont work yet
    B = inv(sqrt(E)) ⊗ Matrix{Int64}(I,dim_out,dim_out)
    C = B * pos_A * B
    return C
end


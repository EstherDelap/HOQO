using LinearAlgebra

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


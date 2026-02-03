include("functions.jl")

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


dim_sys = 2
dim_env = 2
dim_aux = 2
#rho = rand_rho(dim_sys*dim_env) #dim = dim_sys*dim_env (matrix)
a=[1,0]*[1,0]' 
#U= haar_measure(dim_sys*dim_aux) #dim_sys*dim_a



#CODE TO CHECK VEC_UNITARY WORKS

#rho_dashed_normal = vec(U*rho*U')
#rho_dashed_vec = vec_unitary(U) * vec(rho) #this is a vectorised version of rho' = U rho U^dag
#println(isapprox(rho_dashed_normal, rho_dashed_vec))



#CODE TO CHECK APPLY_PARTIAL_UNITARY_VEC WORKS

#rho_1 = apply_partial_unitary_vec(rho, a, U, dim_env) #takes in rho as a matrix on spaces {env, sys}, outputs a vector rho' on spaces {env, sys, aux} 

#rho_2 = vec(apply_partial_unitary(rho, a, U, dim_env))
#println(isapprox( rho_1, rho_2))

#rho_1_tensor = reshape(rho_1,(dim_aux,dim_sys,dim_env,dim_aux,dim_sys,dim_env))



#CODE TO CHECK PROB_B WORKS (at least it returns probabilities that sum to 1)

#probs = prob_b( rho_1_tensor , dim_sys, dim_env, dim_aux) #rho must be a rank 6 tensor 
#println(sum(probs))
#testing_probabilities(reshape(rho_1 , (dim_aux*dim_env*dim_sys, dim_aux*dim_env*dim_sys)), dim_aux, dim_env, dim_sys )

#CODE CHECKING IT WORKS WITH STATES AND UNITARIES THAT WE KNOW


U = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0] #the CNOT gate unitary

#rho = ([1,0]*[1,0]') ⊗ (1/2*[1,1]*[1,1]')
#rho_dashed = reshape(apply_partial_unitary_vec(rho, a, U, dim_env), (dim_aux, dim_sys, dim_env, dim_aux, dim_sys, dim_env))
#rho_dashed = reshape(apply_partial_unitary_vec(rho, a, U, dim_env), (dim_aux*dim_sys*dim_env, dim_aux*dim_sys*dim_env))

#testing_probabilities(rho_dashed, dim_aux, dim_env, dim_sys ) #should produce probabilities 0.5, 0.5

#probs = prob_b(rho_dashed, dim_sys, dim_env, dim_aux ) #should produce probabilities 0.5, 0.5
#rho_final = weighted_rho(rho_dashed, probs, dim_aux, dim_sys, dim_env)  #produces either |00> if b is measured as |0> (i.e. 1 is printed) or |01> if b is measured as |1> (i.e. 2 is printed)


alpha = 0.4 #for psi to be normalised, mist be between 0 and 1
psi = sqrt(alpha ) * [1,0] + sqrt(1-alpha) * [0,1]
rho = (1/2*[1 1; 1 1]) ⊗ (psi * psi') 
rho_dashed = reshape(apply_partial_unitary_vec(rho, a, U, dim_env), (dim_aux, dim_sys, dim_env, dim_aux, dim_sys, dim_env))

#testing_probabilities(rho_dashed, dim_aux, dim_env, dim_sys ) #should prodce probabilities alpha and 1-alpha

probs = prob_b(rho_dashed , dim_sys, dim_env, dim_aux ) #should produce probabilities alpha and 1-alpha
rho_final = weighted_rho(rho_dashed, probs, dim_aux, dim_sys, dim_env) 

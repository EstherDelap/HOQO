include("functions.jl")
include("functions_channel.jl")

function observable_channel_tomography(choi, obs, d, num_rounds; existing_ave_inv = zeros(d^2,d^2), existing_ave_map =zeros(d^2,d^2), prev_n_shots = 0)
    """
    Caluclates tr(channel*(obs_1⊗obs_2)) for both the actual choi matrix of the channel and the 
    shadow of the channel, then returns the difference between them     
    """
    shadow_inv, shadow_map = channel_tomography(choi, d, num_rounds, existing_ave_inv, existing_ave_map, prev_n_shots)
    guess= tr(obs*shadow_inv)
    return guess, shadow_inv, shadow_map
end

function plot_observable(choi, obs, d, total_num_rounds; start = 1)
    """
    For increasing n in total_num_rounds, calculates the shadow of choi then finds the relative error
    between the guessed value of the observable given by the shadow and the actual outcome given by choi.  
    """
    shots= zeros(1)
    rel_err = zeros(0)
    ave_inv = zeros(d^2,d^2)
    ave_map = zeros(d^2,d^2)
 
    actual_value = tr(obs*choi)

    for n=start:total_num_rounds
        num_rounds = 10^n
        append!(shots, num_rounds)
        guess, ave_inv, ave_map = observable_channel_tomography(choi, obs, d, num_rounds-shots[end-1], existing_ave_inv= ave_inv, existing_ave_map=ave_map, prev_n_shots = shots[end-1] )
        append!(rel_err, norm(guess-actual_value)/norm(actual_value))
     
    end
    return deleteat!(shots,1), rel_err
end

function plot_observable_shadow(choi, obs, d, total_num_rounds; start= 1)
    """
    plots the shadow tomography on a channel with an observable versus wihtout an observable.
    Using an observable should make channel tomography work better. 
    """
    shots= zeros(1)
    rel_err_shadow = zeros(0)
    rel_err_obs = zeros(0)
    
    ave_inv = zeros(d^2,d^2)
    ave_map = zeros(d^2,d^2)
 
    actual_value = tr(obs*choi)

    for n=start:total_num_rounds
        num_rounds = 10^n
        append!(shots, num_rounds)
        guess, ave_inv, ave_map = observable_channel_tomography(choi, obs, d, num_rounds-shots[end-1], existing_ave_inv= ave_inv, existing_ave_map=ave_map, prev_n_shots = shots[end-1] )
        append!(rel_err_obs, norm(guess-actual_value)/norm(actual_value)) #relative error between tr(O*shadow) and tr(O*choi)
        append!(rel_err_shadow, norm(choi - ave_inv)/norm(ave_inv) ) #relaative error between shadow and choi
    end
    return deleteat!(shots,1), rel_err_obs, rel_err_shadow
end    


function median_of_means(choi, obs, d, N, K)
    """
    performs median of means shadow tomography of the channel choi and observable obs by forming
    K independent sample means, each sample mean formed by shadow tomography on the choi state with
    the observable N times. The median of the independent sample means is returned.
    """
    collection = zeros(0)
    
    for k=1:K 
        guess = observable_channel_tomography(choi, obs, d, N)[1]
        append!(collection, guess)
    end
    return( median( sort(collection) ) )
end

function plot_median_means(total_num_shots, choi, obs, d, K; start = 1)
    """
    Plots the shadow tomography on channel choi for observable obs using median of means 
    for increasing number of total shots = NK
    """
    shots = zeros(0)
    actual_value = tr(obs*choi)
    rel_err = zeros(0)
    for n=start:total_num_shots
        num_shots = 10^n
        append!(shots, num_shots)
        N = convert(Int64, num_shots/K)
        guess = median_of_means(choi, obs, d, N, K)
        append!(rel_err, norm(guess-actual_value)/norm(actual_value))
    end
    return shots, rel_err 
end


function plot_shadow_obs_dim( total_dim, num_shots)
    """
    Plots the relative error for the shadow and for the observable for increasing dimensions. So for
    each dimension starting at d=2, a random choi matrix is generated and an observable, then shadow 
    tomography is performed on this choi matrix using num_shots, and a guess of the outcome of the 
    observable is found using the shadow. The relative error between the shadow and the choi matrix, and 
    the relative error between the guess of the outcome and the actual outcome are both stored for each
    dimension.  
    """
    dimensions = zeros(0)
    rel_err_shadow = zeros(0)
    rel_err_obs = zeros(0)
    
    for d=2:total_dim
        append!(dimensions, d)
        
        choi = rand_CPTP(d,d)
        obs = rand_rho(d) ⊗ (ket(1,d)*bra(1,d))
        
        actual_value = tr(obs*choi)
        existing_ave = zeros(d^2,d^2)
        
        shadow = channel_tomography(choi,d, num_shots, existing_ave, existing_ave, 0)[1]
        guess = tr(obs*shadow)
        
        append!( rel_err_shadow, norm(choi - shadow)/norm(choi) )
        append!( rel_err_obs, norm(actual_value - guess)/norm(guess) )
    end
    return dimensions, rel_err_shadow, rel_err_obs
end
#d=2
#choi = rand_CPTP(d,d)
#ptrace(choi,[d,d],[2])
#observable = rand_rho(d) ⊗ (ket(1,2)*bra(1,2)) #ordered correctly as input_state ⊗ output measurement

#x,y = plot_observable(choi, observable, 2, 7, start=3)
#plot(x,y,xaxis = :log10, label = "tr(O * shadow) vs tr(O * choi)")

#x, y_obs, y_shadow = plot_observable_shadow(choi, observable, d, 7, start = 3)
#plot(x, [y_obs y_shadow], xaxis = :log10, layout=(2, 1),label = ["tr(O*shadow) vs tr(O*choi)" "shadow vs choi"])

#x,y = plot_median_means(7, choi, observable, d, 100, start = 3)
#plot(x,y, xaxis = :log10)

#x, y_shadow, y_obs = plot_shadow_obs_dim( 6, 10^6)
#plot(x, [y_shadow y_obs], layout = (2,1), label = [" shadow vs choi for increasing dimensions" "guess of observable for increasing dimensions"])


function random_ginibre(n,m)
    return rand(Complex{Float64}, (n,m))
end

function random_haar_povm(d)
    """
    Generates a Haar distributed random POVM for a Hilbert space of dimensions d. It returnes
    k=d^2 POVM elements, which sum to the identity and are positive semidefinite. 
    """
    k=d^2
    povm = Vector{Matrix{ComplexF64}}(undef, 0)
    S = zeros(d,d)
    for i=1:k
        Xi = random_ginibre(d,1)
        Wi = Xi * Xi'
        append!(povm,[Wi])
        S+=Wi
    end
    S = inv(sqrt(S))
    full_povm = [S*M*S for M in povm]
    return full_povm
end

function random_haar_effect(d)
    """
    Generates a Haar distributed random POVM effect of Hilbert space dimensions d, as if it
    were part of a POVM of d^2 elements 
    """
    k=d^2
    X = random_ginibre(d,1)
    W = X*X'
    Y = random_ginibre(d, k-1)
    S = W + (Y*Y')
    S = inv(sqrt(S))
    return S*W*S'
end

function tester_element(d_1, d_2, d_a)
    """
    Generates a random tester element going from dimensions d_1 and auxiliary space d_a to 
    d_2 with auxilliary space d_a. 
    """
    rho = rand_rho(d_1*d_a)
    E = random_haar_effect(d_2*d_a)
    in_a_out =  (Matrix{Int64}(I,d_1,d_1) ⊗ transpose(E))*( rho ⊗ Matrix{Int64}(I,d_2,d_2))
    return ptrace(in_a_out, [d_1, d_a, d_2],[2]) #i think it is ordered like this
end

#T = tester_element(d,d,2)
#x,y = plot_observable(choi, T, 2, 7, start=3)
#plot(x,y,xaxis = :log10, label = "tr(T * shadow) vs tr(T * choi)")

function plot_observable_tester(choi, obs, tester, total_num_rounds; start = 1)
    shots= zeros(1)
    rel_err_obs = zeros(0)
    rel_err_tester = zeros(0)

    ave_inv_o = zeros(d^2,d^2)
    ave_map_o = zeros(d^2,d^2)
    ave_inv_t = zeros(d^2,d^2)
    ave_map_t = zeros(d^2,d^2)
 
    value_obs = tr(obs*choi)
    value_tester = tr(tester*choi)

    for n=start:total_num_rounds
        num_rounds = 10^n
        append!(shots, num_rounds)
        guess_o, ave_inv_o, ave_map_o = observable_channel_tomography(choi, obs, d, num_rounds-shots[end-1], existing_ave_inv= ave_inv_o, existing_ave_map=ave_map_o, prev_n_shots = shots[end-1] )
        guess_t, ave_inv_t, ave_map_t = observable_channel_tomography(choi, tester, d, num_rounds-shots[end-1], existing_ave_inv= ave_inv_t, existing_ave_map=ave_map_t, prev_n_shots = shots[end-1] )
        append!(rel_err_obs, norm(guess_o-value_obs)/norm(value_obs))
        append!(rel_err_tester, norm(guess_t-value_tester)/norm(value_tester))
    end
    return deleteat!(shots,1), rel_err_obs, rel_err_tester
end

#x, y_1, y_2 = plot_observable_tester(choi, observable, T, 7, start = 3)
#plot(x, [y_1 y_2] , xaxis = :log10, label = ["observable" "tester"])


#maximally entangled state and bell measurement
#phi_plus = max_entangled(d*2)
#phi_state = proj(phi_plus)
#T = ptrace(((Matrix{Int64}(I,d,d) ⊗ transpose(phi_state))*( phi_state ⊗ Matrix{Int64}(I,d,d))), [d,2,d], [2])
#x,y = plot_observable(choi, T, 2, 7, start=3)
#plot(x,y,xaxis = :log10, label = "tr(T * shadow) vs tr(T * choi) for maximally entangled state")
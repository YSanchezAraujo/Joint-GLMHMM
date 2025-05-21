include("model_utils.jl"));

using LogExpFunctions;
using Distributions;
using LinearAlgebra;
using Random;
using Clustering;
using GLM;
using StatsBase;
using Optim;
using ProgressBars;
using ToeplitzMatrices;
using FFTW;
using BandedMatrices;


"""
    Compute the log-likelihood of the data under the Gaussian GLM-HMM model with the covariance kernel.

    Parameters:
    - loglik_obs_gauss: Matrix to store the log-likelihoods.
    - X: Matrix representing the design matrix.
    - Y: Matrix representing the response data, n_steps by 3
    - W: 3-D of array, each representing the weight matrix for a state.
    - sigma: Vector of length n_states, each representing the standard deviation of the data
    - rho: Vector of length n_states, each representing the correlation coefficient of the data
    - n_steps: Number of steps in the data.
    - n_states: Number of states in the model.
"""
function compute_gauss_loglikelihoods!(
    loglik_obs_gauss::Matrix, 
    X::Any, 
    Y::Vector{Matrix{Float64}}, 
    W::Array, 
    Sigma::Array,
    n_steps::Int, 
    n_states::Int
)::Nothing
    D = size(Y[1], 2)
    z = zeros(D)
    for t in 1:n_steps
        for s in 1:n_states
            n = size(Y[t], 1)
            residuals = Y[t] .- X[t] * W[:, :, s]
            dist = MvNormal(z, Sigma[:, :, s])
            ll = [logpdf(dist, residuals[i, :]) for i in 1:n]
            loglik_obs_gauss[t, s] = sum(ll) / (D * n)
        end
    end
    nothing
end



"""
    Compute the log-likelihood of the data under the Bernoulli GLM-HMM model.

    Parameters:
    - loglik_obs_bern: Matrix to store the log-likelihoods.
    - X: Matrix representing the design matrix.
    - y: Vector representing the response data.
    - W: 3-D of array, each representing the weight matrix for a state
    - n_steps: Number of steps in the data.
    - n_class: Number of classes in the model.
    - n_states: Number of states in the model.
"""
function compute_bern_loglikelihoods!(
    loglik_obs_bern::Matrix, 
    X::Matrix, 
    y::Vector, 
    W::Array{Float64,2}, 
    n_steps::Int, 
    n_class::Int, 
    n_states::Int
)::Nothing

    Y = one_hot(y)'
    XW = zeros(n_steps, n_class, n_states);
    XW[:, 2, :] = X * W;
    logpy_bern = XW .- logsumexp(XW; dims=2);
    loglik_obs_bern .= drop_dim(sum(logpy_bern .* Y'; dims=2))
    nothing
end


struct log_forward
    log_alpha::Matrix
    log_c::Vector
end

struct log_backward
    log_beta::Matrix
end

struct prob_param
    A::Array
    pi0::Vector
    dir_prior::Matrix 
end


function log_normalize(log_prob::Vector)
    log_c = logsumexp(log_prob)
    return log_prob .- log_c, log_c
end


function log_forward_message!(
    log_f_obj::log_forward, 
    log_lik_obs::Matrix, 
    log_pi0::Vector, 
    log_A::Matrix; 
    first_row::Int = 1, 
    last_row::Int = size(log_lik_obs, 1)
)
    n_steps = size(log_lik_obs, 1)
    k = size(log_lik_obs, 2)
    rows = first_row:last_row
    log_alpha_t, log_c_t = log_normalize(log_lik_obs[1, :] .+ log_pi0)
    log_f_obj.log_alpha[rows[1], :] .= log_alpha_t
    log_f_obj.log_c[rows[1]] = log_c_t

    for t in 2:n_steps
        log_alpha_t, log_c_t = log_normalize(log_lik_obs[t, :] .+ vec(logsumexp(log_A .+ log_f_obj.log_alpha[rows[t-1], :]; dims=1)))
        log_f_obj.log_alpha[rows[t], :] .= log_alpha_t
        log_f_obj.log_c[rows[t]] = log_c_t
    end

end

function log_backward_message!(
    log_b_obj::log_backward, 
    log_lik_obs::Matrix, 
    log_A::Matrix, 
    log_c::Vector; 
    first_row::Int = 1, 
    last_row::Int = size(log_lik_obs, 1)
)
    rows = first_row:last_row
    log_b_obj.log_beta[rows[end], :] .= 0.0
    for t in size(log_lik_obs, 1)-1:-1:1
        log_b_obj.log_beta[rows[t], :] .= (
            vec(logsumexp(log_A' .+ (log_b_obj.log_beta[rows[t+1], :] .+ log_lik_obs[t+1, :]); dims=1)) .- log_c[t+1]
        )
    end
end

function expectations(log_f_obj::log_forward, log_b_obj::log_backward, log_lik_obs::Matrix, log_A::Matrix, session_boundaries::Vector)
    n_states = size(log_lik_obs, 2)
    n_obs = size(log_lik_obs, 1)
    log_xi = fill(-Inf,n_obs-1, n_states, n_states)

    log_gamma = log_f_obj.log_alpha .+ log_b_obj.log_beta
    log_gamma .-= logsumexp(log_gamma, dims=2)
    gamma = exp.(log_gamma)

    for (start_idx, end_idx) in session_boundaries
        # fill xi for t = start_idx:(end_idx - 1)
        for t in start_idx:(end_idx - 1)
            log_b_lik = log_lik_obs[t+1, :] .+ log_b_obj.log_beta[t+1, :]
            log_xi[t, :, :] .= (log_A .+ (log_f_obj.log_alpha[t, :] .+ log_b_lik')) .- log_f_obj.log_c[t+1]
        end
    end

    xi = exp.(log_xi)
    xi_sum = drop_dim(sum(xi, dims=1))

    return (
        xi = xi_sum,
        gamma = gamma,
        pi0 = gamma[1, :]
    )
end


function prob_param_m_step!(prob_obj, e_quants, first_last_inds)
    pi0_sum = zeros(size(prob_obj.pi0))
    for (start_idx, _) in first_last_inds
        pi0_sum .+= e_quants.gamma[start_idx, :]
    end
    prob_obj.pi0 .= pi0_sum ./ sum(pi0_sum) 

    for i in 1:size(prob_obj.A, 1)
        N_i = e_quants.xi[i, :] .+ prob_obj.dir_prior[i, :]
        prob_obj.A[i, :] .= N_i ./ sum(N_i)
    end
end




function bern_gradient_neg_loglik_with_prior!(G, w, X, y, gamma)
    mu = X * w
    sigma_mu = 1 ./ (1 .+ exp.(-mu))
    G .= - X' * (gamma .* y) + X' * (gamma .* sigma_mu)
    G .+= w
end

function bern_hessian_neg_loglik_with_prior!(H, w, X, y, gamma)
    mu = X * w
    sigma_mu = 1 ./ (1 .+ exp.(-mu))  
    sigma_prime_mu = sigma_mu .* (1 .- sigma_mu)
    D = Diagonal(gamma .* sigma_prime_mu)
    H .= X' * D * X
    
    for i in 1:size(H,1)
        H[i,i] += 1
    end
end

function bern_neg_loglik_with_prior(w, X, y, gamma_vec)
    mu = X * w
    nll = - (gamma_vec .* y)' * mu + sum(gamma_vec .* softplus.(mu))
    nll += 0.5 * sum(w.^2)
    return nll
end


function bern_m_step_with_derivs(X, y, gamma, n_cols, n_states)
    W = fill(NaN, n_cols, n_states)

    for i in 1:n_states
        gamma_state = gamma[:, i]
        opts = optimize(
            w -> bern_neg_loglik_with_prior(w, X, y, gamma_state),
            (G, w) -> bern_gradient_neg_loglik_with_prior!(G, w, X, y, gamma_state),
            (H, w) -> bern_hessian_neg_loglik_with_prior!(H, w, X, y, gamma_state),
            zeros(n_cols),
            NewtonTrustRegion();
        )
        W[:, i] = opts.minimizer
    end

    return W
end




function batch_ridge_regression(
    X::Any, 
    Y::Vector{Matrix{Float64}},
    ridge_lambda_squared::Real
)::Matrix

    p = size(X[1], 2)
    fxx(x) = x'x
    fxy(x, y) = x'y

    S_xx = mapreduce(fxx, +, X) + ridge_lambda_squared * I(p) 
    S_xy = mapreduce(fxy, +, X, Y)
    
    W = S_xx \ S_xy  
    
    return W
end

function batch_ridge_regression(
    X::Any, 
    Y::Vector{Matrix{Float64}}, 
    gamma_list::Any, 
    k::Int,
    ridge_lambda_squared::Real
)::Matrix{Float64}

    p = size(X[1], 2)
    M = size(Y[1], 2)

    S_xx = zeros(p, p)
    S_xy = zeros(p, M)

    for i in 1:length(X)
        sqrt_gamma = sqrt.(gamma_list[i])[:, k]
        Xz = sqrt_gamma .* X[i]
        Yz = sqrt_gamma .* Y[i]
        S_xx .+= Xz' * Xz
        S_xy .+= Xz' * Yz
    end

    S_xx .+= ridge_lambda_squared * I(p)
    W = S_xx \ S_xy 

    return W
end

function compute_inds(n_samp_per_t::Vector)
    last_ind = cumsum(n_samp_per_t)
    first_ind = [1; last_ind[1:end-1] .+ 1]
    return first_ind, last_ind
end


function batch_full_covariance(
    W::Matrix{Float64},
    X::Any,
    Y::Vector{Matrix{Float64}},
)
    M = size(W, 2)
    sum_of_rrt = zeros(M, M)
    n = sum(size.(X, 1))

    for i in 1:length(X)
        mu_i = X[i] * W
        resid_i = Y[i] .- mu_i
        sum_of_rrt .+= resid_i' * resid_i
    end

    Sigma = sum_of_rrt / n
    Sigma = 0.5 * (Sigma + Sigma')

    return Symmetric(Sigma)
end

function batch_full_covariance(
    W::Matrix{Float64},
    X::Any,
    Y::Vector{Matrix{Float64}},
    gamma_list::Any,
    k::Int
)
    M = size(W, 2)
    sum_of_rrt = zeros(M, M)
    sum_of_gamma = 0.0

    for i in 1:length(X)
        sqrt_gamma = sqrt.(gamma_list[i])[:, k]
        mu_i = X[i] * W
        resid_i = sqrt_gamma .* (Y[i] .- mu_i)
        sum_of_rrt .+= resid_i' * resid_i
        sum_of_gamma += sum(gamma_list[i][:, k])
    end

    Sigma = sum_of_rrt / sum_of_gamma
    Sigma = 0.5 * (Sigma + Sigma')

    return Symmetric(Sigma)
end

function gamma_into_list(
    gamma::Matrix{Float64}, 
    inds::Vector{Tuple{Int64, Int64}},
    n_samp_per_trial::Vector{Int64}
)
    gamma_list = []
    for i in 1:length(inds)
        z = n_samp_per_trial[inds[i][1]:inds[i][2]]
        big_gamma = vcat(_repeat_by_trial_length(z, gamma[inds[i][1]:inds[i][2], :])...)
        push!(gamma_list, big_gamma)
    end

    return gamma_list
end

function batch_gauss_m_step_full_covariance(
    X::Any,
    Y::Vector{Matrix{Float64}},   
    gamma::Matrix{Float64},
    inds::Vector{Tuple{Int64, Int64}},
    n_samp_per_trial::Vector{Int64},
    ridge_lambda_squared::Real
)

    K = size(gamma, 2)
    p = size(X[1], 2)
    M = size(Y[1], 2)

    W = zeros(p, M, K)
    Sigma = zeros(M, M, K)
    gamma_list = gamma_into_list(gamma, inds, n_samp_per_trial)

    for k in 1:K
        W[:, :, k] = batch_ridge_regression(
            X,      
            Y,      
            gamma_list,
            k,
            ridge_lambda_squared
        )

        Sigma[:, :, k] = batch_full_covariance(
            W[:, :, k],
            X,
            Y,
            gamma_list,
            k,
        )
    end

    return W, Sigma
end


function fit_gaussbern_glmhmm_with_em_full_covar(
    X_bern, 
    X_gauss_trial, 
    X_gauss_ses,
    y_bern, 
    Y_gauss_trial, 
    Y_gauss_ses,
    n_per_ses,
    n_states,
    dir_prior_diag,
    dir_prior_off_diag,
    ridge_lambda_squared,
    ;
    model_init = nothing,
    init_type = nothing,
    rng_num = 9998,
    tol = 1e-3,
    max_iter = 150,
    n_class = 2
    )

    n_trials_total = size(X_bern, 1)
    n_cols_bern = size(X_bern, 2)
    n_samp_per_trial = size.(Y_gauss_trial, 1)
    rng_state = isnothing(rng_num) ? nothing : MersenneTwister(rng_num)

    # these are for the gaussian m-step, to expand the gammas 
    last_inds = cumsum(n_per_ses)
    first_last_inds = collect(zip([1; last_inds[1:end-1] .+ 1], last_inds))
    Sigma_init = zeros(size(Y_gauss_ses[1], 2), size(Y_gauss_ses[1], 2), n_states)
    g_init = batch_ridge_regression(X_gauss_ses, Y_gauss_ses, ridge_lambda_squared) .+ rand(rng_state, size(X_gauss_ses[1], 2), size(Y_gauss_ses[1], 2), n_states) * 0.1
    dir_prior = generate_dir_prior(n_states, dir_prior_diag, dir_prior_off_diag)

    for k in 1:n_states
        Sigma_init[:, :, k] .= batch_full_covariance(g_init[:, :, k], X_gauss_ses, Y_gauss_ses)
    end

    prob_obj = prob_param(
        init_A_from_prior(rng_state, n_states, dir_prior),
        [1 / n_states for n in 1:n_states],
        dir_prior
    )

    if isnothing(model_init) && isnothing(init_type)
        b_init = init_bern(rng_state, X_bern, y_bern, n_states)

        model_init = (
            W_bern = b_init,
            W_gauss = g_init,
            Sigma = Sigma_init,
            dir_prior = dir_prior,
            prob_obj = prob_obj
        )
    elseif init_type == "bernoulli"
        model_init = (
            W_bern = copy(model_init.W_bern),
            W_gauss = g_init,
            Sigma = Sigma_init,
            dir_prior = dir_prior,
            prob_obj = prob_obj
        )
    elseif init_type == "gauss_bern"
        model_init = (
            W_bern = copy(model_init.W_bern),
            W_gauss = copy(model_init.W_gauss),
            Sigma = copy(model_init.Sigma),
            dir_prior = dir_prior,
            prob_obj = model_init.prob_obj
        )
    end

    W_bern = copy(model_init.W_bern);
    W_gauss = copy(model_init.W_gauss);
    Sigma = copy(model_init.Sigma);
    dir_prior = copy(model_init.dir_prior);
    prob_obj = prob_param(
        copy(model_init.prob_obj.A),
        copy(model_init.prob_obj.pi0),
        dir_prior
    )

    log_marg_lik = fill(NaN, max_iter+1)

    # Initialize log-likelihood matrices
    loglik_obs_bern = zeros(n_trials_total, n_states)
    loglik_obs_gauss = zeros(n_trials_total, n_states)
    log_lik_obs = zeros(n_trials_total, n_states)

    log_f_msg = log_forward(zeros(n_trials_total, n_states), zeros(n_trials_total))
    log_b_msg = log_backward(zeros(n_trials_total, n_states))

    compute_bern_loglikelihoods!(loglik_obs_bern, X_bern, y_bern, W_bern, n_trials_total, n_class, n_states)
    compute_gauss_loglikelihoods!(loglik_obs_gauss, X_gauss_trial, Y_gauss_trial, W_gauss, Sigma, n_trials_total, n_states)

    log_lik_obs .= loglik_obs_bern .+ loglik_obs_gauss

    for (a, b) in first_last_inds
        log_forward_message!(log_f_msg, log_lik_obs[a:b, :], log.(prob_obj.pi0), log.(prob_obj.A); first_row = a, last_row = b)
        log_backward_message!(log_b_msg, log_lik_obs[a:b, :], log.(prob_obj.A), log_f_msg.log_c[a:b]; first_row = a, last_row = b)
    end

    e_quants = expectations(log_f_msg, log_b_msg, log_lik_obs, log.(prob_obj.A), first_last_inds)
    prob_param_m_step!(prob_obj, e_quants, first_last_inds)

    W_bern = bern_m_step_with_derivs(X_bern, y_bern, e_quants.gamma, n_cols_bern, n_states)

    W_gauss, Sigma = batch_gauss_m_step_full_covariance(X_gauss_ses, Y_gauss_ses, e_quants.gamma, first_last_inds, n_samp_per_trial, ridge_lambda_squared)

    lml_prev = -Inf
    converged = false
    iteration_counter = 1

    for iter in ProgressBar(1:max_iter)
        compute_bern_loglikelihoods!(loglik_obs_bern, X_bern, y_bern, W_bern, n_trials_total, n_class, n_states)
        compute_gauss_loglikelihoods!(loglik_obs_gauss, X_gauss_trial, Y_gauss_trial, W_gauss, Sigma, n_trials_total, n_states)
    
        log_lik_obs .= loglik_obs_bern .+ loglik_obs_gauss
    
        for  (a, b) in first_last_inds
            log_forward_message!(log_f_msg, log_lik_obs[a:b, :], log.(prob_obj.pi0), log.(prob_obj.A); first_row = a, last_row = b)
            log_backward_message!(log_b_msg, log_lik_obs[a:b, :], log.(prob_obj.A), log_f_msg.log_c[a:b]; first_row = a, last_row = b)
        end
    
        e_quants = expectations(log_f_msg, log_b_msg, log_lik_obs, log.(prob_obj.A), first_last_inds)
        prob_param_m_step!(prob_obj, e_quants, first_last_inds)
    
        W_bern = bern_m_step_with_derivs(X_bern, y_bern, e_quants.gamma, n_cols_bern, n_states)

        W_gauss, Sigma = batch_gauss_m_step_full_covariance(X_gauss_ses, Y_gauss_ses, e_quants.gamma, first_last_inds, n_samp_per_trial, ridge_lambda_squared)

        lml = sum(log_f_msg.log_c)
        log_marg_lik[iter] = lml

        if abs(lml - lml_prev) < tol
            converged = true
            break
        end

        lml_prev = lml
        iteration_counter += 1
    end

    return (
        W_bern = W_bern,
        W_gauss = W_gauss,
        Sigma = Sigma,
        gamma = e_quants.gamma,
        A = prob_obj.A,
        prob_obj = prob_obj,
        e_quants = e_quants,
        lml = log_marg_lik[1:iteration_counter],
        converged = converged,
        rng_state = rng_num,
        ll_bern = loglik_obs_bern,
        ll_gauss = loglik_obs_gauss,
    )
end

function get_best_model(files::Vector{String})
    best_model = nothing
    best_lml = -Inf
    best_file = nothing

    for f in files
        model = load(f, "model")
        if model.lml[end] > best_lml
            best_model = model
            best_lml = model.lml[end]
            best_file = f
        end
    end

    println("Best model found in file: $best_file")

    return best_model
end


function load_best_model_B(n_state::Int, load_dir::String)
    load_file = "$(n_state)_state_bernoulli_megamouse_rngnum_"
    files = [joinpath(load_dir, f) for f in readdir(load_dir) if startswith(f, load_file)]
    return get_best_model(files)
end

function load_best_model_GB(n_state::Int, load_dir::String)
    load_file = "$(n_state)_state_gauss_bern_megamouse_rngnum_"
    files = [joinpath(load_dir, f) for f in readdir(load_dir) if startswith(f, load_file)]
    return get_best_model(files)
end

function get_best_model_indiv(files::Vector{String})
    best_model = nothing
    best_lml = -Inf
    best_file = nothing

    for f in files
        model = load(f, "model_fit")
        if model.lml[end] > best_lml
            best_model = model
            best_lml = model.lml[end]
            best_file = f
        end
    end

    println("Best model found in file: $best_file")

    return best_model
end


function load_best_model_indiv(n_state::Int, mouse_id::Int, load_dir::String)
    load_file = "_n_state_$(n_state)_mouse_$(mouse_id).jld2"
    files = [joinpath(load_dir, f) for f in readdir(load_dir) if endswith(f, load_file)]
    return get_best_model_indiv(files)
end


function count_changes(state_seq::Vector{Int})
    n_changes = 0
    for i in 1:length(state_seq) - 1
        if state_seq[i] != state_seq[i + 1]
            n_changes += 1
        end
    end
    return n_changes
end


function rank1_svd_decomp(weight_set)
    U, s, V = svd(weight_set)
    time_kernel = U[:, 1] * s[1]
    scale_kernel = V[:, 1]

    final_sign = all(sign.(scale_kernel) .== -1) ? -1 : 1
    scale_factor = s[1]
    shapes = weight_set / scale_factor
    var_exp = s[1]^2 / sum(s.^2)

    return (
        recon = time_kernel * scale_kernel',
        time_kernel = time_kernel * final_sign,
        scale_kernel = scale_kernel * final_sign,
	    scale = scale_factor,
	    shapes = shapes,
	    var_exp = var_exp
    )
end

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Tuple, List, NamedTuple
from jax import jit, lax, nn
import optax
from functools import partial
from tqdm import tqdm # For the progress bar
from collections import namedtuple
from jax.scipy.stats import multivariate_normal as jax_mvnormal

@jit
def log_normalize(log_prob: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalizes log probabilities
    
    Args:
        log_prob: A vector of log probabilities
    
    Returns:
        A tuple containing the normalized log probabilities and the log of the normalization constant.
    """
    log_c = logsumexp(log_prob)
    return log_prob - log_c, log_c

@jit
def compute_log_forward_message(
    log_lik_obs: jnp.ndarray,
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    Computes the forward messages for a Hidden Markov Model.
    
    Args:
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_pi0: Log initial state probabilities, shape (n_states,).
        log_A: Log transition matrix, shape (n_states, n_states).
    
    Returns:
        A tuple containing log_alpha, and the log normalizers.
    """
    n_steps, _ = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_alpha,  = carry
        log_alpha_step, log_c_step = log_normalize(
            log_lik_obs[step, :] + logsumexp(log_A + prev_log_alpha[:, jnp.newaxis], axis=0)
            )
        return (log_alpha_step, ), (log_alpha_step, log_c_step)

    initial_log_alpha, initial_log_c = log_normalize(log_lik_obs[0, :] + log_pi0)
    initial_carry = (initial_log_alpha, )

    _, scan_output = lax.scan(scan_step, initial_carry, jnp.arange(1, n_steps))

    log_alpha, log_c = scan_output
    log_alpha = jnp.vstack([initial_log_alpha, log_alpha])
    log_c = jnp.hstack([initial_log_c, log_c])

    return log_alpha, log_c


@jit
def compute_log_backward_message(
    log_lik_obs: jnp.ndarray, 
    log_A: jnp.ndarray, 
    log_c: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the backward messages for a Hidden Markov Model.
    
    Args:
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_A: Log transition matrix, shape (n_states, n_states).
        log_c: Log normalization constants from forward messages, shape (n_steps,).
    
    Returns:
        Log beta messages.
    """
    n_steps, n_states = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_beta, = carry
        log_beta_sum = prev_log_beta + log_lik_obs[step+1, :]
        log_beta_step = logsumexp(log_A.T + log_beta_sum[:, jnp.newaxis], axis=0) - log_c[step+1]
        return (log_beta_step, ), log_beta_step

    initial_log_beta = jnp.zeros(n_states) 
    initial_carry = (initial_log_beta, )
    _, scan_output = lax.scan(scan_step, initial_carry, jnp.arange(n_steps-2, -1, -1))
    log_beta = jnp.vstack([jnp.flip(scan_output, axis=0), initial_log_beta])

    return log_beta

@jax.jit
def compute_expectations(
    log_alpha: jnp.ndarray,
    log_beta: jnp.ndarray,
    log_c: jnp.ndarray,
    log_lik_obs: jnp.ndarray,
    log_A: jnp.ndarray   
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the expectations (xi and gamma) for the Hidden Markov Model.

    Args:
        log_alpha: Log scaled forward messages from compute_log_forward_message.
        log_beta: Log scaled backward messages from compute_log_backward_message.
        log_c: Log normalization constants from forward messages.
        log_lik_obs: Log likelihoods of observations.
        log_A: Log transition matrix.

    Returns:
        A tuple containing:
            - 'xi_summed': Expected transitions sum_t xi_t(i,j),
                           shape (n_states, n_states). If not transposed, xi_summed[i,j]
                           is the expected number of transitions from state i to state j.
                           The original code had a .T, which is kept here.
            - 'gamma': Expected states P(z_t=i | O), shape (n_steps, n_states).
    """
    n_steps, _ = log_lik_obs.shape

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)
    
    def compute_xi_step(step: int) -> jnp.ndarray:
        log_b_lik = (log_lik_obs[step + 1, :] + log_beta[step + 1, :])[jnp.newaxis, :]
        log_xi_step_ij = log_alpha[step, :][:, jnp.newaxis] + log_A + log_b_lik - log_c[step + 1]
        return jnp.exp(log_xi_step_ij)

    xi_over_time = jax.vmap(compute_xi_step)(jnp.arange(n_steps - 1))
    xi_summed = jnp.sum(xi_over_time, axis=0)

    return xi_summed, gamma


@jit
def bernoulli_glmhmm_loglikelihood(
    X: jnp.ndarray,    
    y: jnp.ndarray,
    W: jnp.ndarray     
) -> jnp.ndarray:
    """
    Computes the log likelihood of the data under the parameters W

    Args:
        X: Design matrix, shape (n_trials, n_features)
        y: Response data (binary 0 or 1), shape (n_trials,)
        W: Weight matrix, shape (n_features, n_states)

    Returns:
        Log likelihood of the data, shape (n_trials, n_states)
    """
    
    if W.ndim == 1:
        _W = W[:, jnp.newaxis]
    elif W.ndim == 2:
        _W = W
    else:
        raise ValueError(f"W must be 1D or 2D, but got {W.ndim} dimensions.")

    logits_per_state = X @ _W
    log_prob_y_eq_1 = nn.log_sigmoid(logits_per_state)
    log_prob_y_eq_0 = nn.log_sigmoid(-logits_per_state)
    y_col = y.astype(jnp.float32)[:, jnp.newaxis]
    loglik_obs_bern= y_col * log_prob_y_eq_1 + (1 - y_col) * log_prob_y_eq_0

    return loglik_obs_bern


@jit
def bern_neg_loglik_with_prior(
    w_bern_state: jnp.ndarray, 
    X_bern: jnp.ndarray,  
    y_bern: jnp.ndarray, 
    gamma_state: jnp.ndarray 
) -> jnp.ndarray:
    """
    Computes the negative log-likelihood for a Bernoulli GLM for a single state,
    including an L2 prior on the weights.

    Args:
        w_bern_state: Weight vector for the current HMM state.
        X_bern: Design matrix for Bernoulli emissions.
        y_bern: Binary response vector (0 or 1).
        gamma_state: Responsibilities (gammas) for the current HMM state.

    Returns:
        Scalar negative log-likelihood value.
    """
    mu = X_bern @ w_bern_state 
    y_bern_float = y_bern.astype(jnp.float32)
    
    term1 = -jnp.dot(gamma_state * y_bern_float, mu)
    term2 = jnp.sum(gamma_state * nn.softplus(mu))
    nll_data = term1 + term2
    
    l2_prior = 0.5 * jnp.sum(w_bern_state**2)
    
    total_nll = nll_data + l2_prior
    return total_nll/X_bern.shape[0] # for stability because of optax, just incase



def optimize_single_state_weights(
    w_initial_s: jnp.ndarray, 
    gamma_s: jnp.ndarray,
    X_bern: jnp.ndarray, 
    y_bern: jnp.ndarray,
    learning_rate: float = 1e-3, 
    num_opt_steps: int = 100    
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimizes weights for a single state in the GLMHMM m-step

    Args:
        w_initial_s: Initial weight vector for the current HMM state.
        gamma_s: Responsibilities (gammas) for the current HMM state.
        X_bern: Design matrix for bernoulli observations.
        y_bern: Binary response vector (0 or 1).
        learning_rate: Learning rate for the optimizer.
        num_opt_steps: Number of optimization steps.

    Returns:
        Estimated parameter vector, and the final loss from the optimizer
    """
    
    loss_fn_s = partial(
        bern_neg_loglik_with_prior, X_bern=X_bern, y_bern=y_bern, gamma_state=gamma_s
    )
    value_and_grad_fn = jax.value_and_grad(loss_fn_s)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state_init = optimizer.init(w_initial_s)
    params_init = w_initial_s

    @jit
    def opt_step(carry, _):
        params_carry, opt_state_carry = carry
        loss_val, grads = value_and_grad_fn(params_carry)
        updates, new_opt_state = optimizer.update(grads, opt_state_carry, params_carry)
        new_params = optax.apply_updates(params_carry, updates)
        return (new_params, new_opt_state), loss_val

    (final_params, final_opt_state), losses = lax.scan(
        opt_step,
        (params_init, opt_state_init),
        None,
        length=num_opt_steps
    )
    
    final_loss = losses[-1] 
    return final_params, final_loss


@partial(jax.jit, static_argnames=("learning_rate", "num_opt_steps")) 
def bern_m_step_jax( 
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    gamma_all_states: jnp.ndarray, 
    initial_W_bern: jnp.ndarray,
    learning_rate: float = 1e-3,
    num_opt_steps: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    Optimizes weights for a all states in the GLMHMM m-step

    Args:
        X_bern: Design matrix for bernoulli observations.
        y_bern: Binary response vector (0 or 1).
        gamma_all_states: Responsibilities (gammas) for all states.
        initial_W_bern: Initial weight vector for the current HMM state.
        learning_rate: Learning rate for the optimizer.
        num_opt_steps: Number of optimization steps.

    Returns:
        Estimated parameter vectors, and the final losses from the optimizer
    """
    
    partial_optimizer = partial(optimize_single_state_weights,
                                 learning_rate=learning_rate,
                                 num_opt_steps=num_opt_steps)
                                 
    optimizer = jax.vmap(
        partial_optimizer, in_axes=(0, 0, None, None) 
    )

    optimized_Ws_T, final_losses_per_state = optimizer(
        initial_W_bern.T, gamma_all_states.T, X_bern, y_bern
    )

    return optimized_Ws_T.T, final_losses_per_state


@partial(jax.jit, static_argnames=("learning_rate", "num_opt_steps"))
def bern_init_opt(
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    learning_rate: float = 1e-3, 
    num_opt_steps: int = 200    
): 
    """
    Optimizes weights for a simple Bernoulli GLM (logistic regression)
    with L2 prior and scaled loss, starting from zero weights, using optax.adam.

    Args:
        X_bern: Design matrix for bernoulli observations.
        y_bern: Binary response vector (0 or 1).        
        learning_rate: Learning rate for the optimizer.
        num_opt_steps: Number of optimization steps.


    Returns:
    """
    gamma_state_ones = jnp.ones(X_bern.shape[0], dtype=X_bern.dtype)

    loss_fn = partial(bern_neg_loglik_with_prior, 
                      X_bern=X_bern,
                      y_bern=y_bern,
                      gamma_state=gamma_state_ones)
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    initial_w = jnp.zeros(X_bern.shape[1], dtype=X_bern.dtype)
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state_init = optimizer.init(initial_w)
    params_init = initial_w

    @jit
    def opt_step_init(carry, _):
        params_carry, opt_state_carry = carry
        loss_val, grads = value_and_grad_fn(params_carry)
        grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), grads)

        updates, new_opt_state = optimizer.update(grads, opt_state_carry, params_carry)
        new_params = optax.apply_updates(params_carry, updates)
        return (new_params, new_opt_state), loss_val
        
    (final_params, _), losses = lax.scan(
        opt_step_init,
        (params_init, opt_state_init),
        None, 
        length=num_opt_steps
    )
    final_loss = losses[-1]
    
    initial_loss = loss_fn(initial_w)
    success_heuristic = (final_loss < initial_loss - 1e-3) & \
                        (~jnp.isnan(final_loss)) & \
                        (~jnp.isinf(final_loss))

    return final_params, success_heuristic.astype(jnp.bool_)


def fit_bern_glmhmm(
    x_set: List[jnp.ndarray],
    y_set: List[jnp.ndarray],
    n_states: int, 
    dirichlet_prior: jnp.ndarray,
    seed: int,
    max_iter = 500,
    em_tol = 1e-3
) -> NamedTuple:
    lml, batch_size = [], len(x_set)
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    lml_prev = -jnp.inf

    x_concat = jnp.concatenate(x_set, axis=0)
    y_concat = jnp.concatenate(y_set, axis=0)

    num_rows = jnp.array([x.shape[0] for x in x_set])
    last_ii = jnp.cumsum(num_rows)
    first_ii = last_ii - num_rows
    batch_size = len(num_rows)

    W, _ = bern_init_opt(x_concat, y_concat)
    n_features = W.shape[0]
    W = (W + random.multivariate_normal(
        subkey, jnp.zeros(n_features),  jnp.diag(jnp.ones(n_features)), (n_states,)
    )).T

    A = jnp.ones((n_states, n_states))
    A = A + dirichlet_prior
    A = A / jnp.sum(A, axis=1, keepdims=True)
    log_A = jnp.log(A)
    pi0 = jnp.ones(n_states) / n_states
    log_pi0 = jnp.log(pi0)
    gamma = jnp.zeros((sum(num_rows), n_states))

    print("Starting EM iterations...")
    for k in tqdm(range(max_iter), desc="EM Iteration"):        
        xi_total = jnp.zeros((n_states, n_states))
        gamma_set = []
        pi0_total = jnp.zeros(n_states)
        lml_total = 0.0
    
        # E-step
        for i in range(batch_size):
            ll_bern = bernoulli_glmhmm_loglikelihood(x_set[i], y_set[i], W)
            log_alpha, log_c = compute_log_forward_message(ll_bern, log_pi0, log_A)
            log_beta = compute_log_backward_message(ll_bern, log_A, log_c)
            xi_i, gamma_i = compute_expectations(log_alpha, log_beta, log_c, ll_bern, log_A)
            xi_total += xi_i
            gamma_set.append(gamma_i)
            pi0_total += gamma_i[0, :]
            lml_total += jnp.sum(log_c)

        # M-step
        A_numerator = xi_total + dirichlet_prior
        A_denominator = jnp.sum(A_numerator, axis=1, keepdims=True)
        A = A_numerator / A_denominator
        log_A = jnp.log(A)

        pi0 = pi0_total / jnp.sum(pi0_total)
        log_pi0 = jnp.log(pi0)

        gamma = jnp.concatenate(gamma_set, axis=0)
        W_new, _ = bern_m_step_jax(x_concat, y_concat, gamma, W)
        W = W_new

        lml.append(lml_total)

        if jnp.abs(lml_prev - lml_total) < em_tol:
            print(f"converged in {k} iterations")
            break

        lml_prev = lml_total

    output = namedtuple(
        'glmhmm', ["gamma", "A", "pi0", "W", "lml", "a", "b"]
    )(gamma=gamma, A=A, pi0=pi0, W=W, lml=lml, a=first_ii, b=last_ii)

    return output

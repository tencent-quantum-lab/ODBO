"""TuRBO algorithm"""

from dataclasses import dataclass
import torch
import math
import numpy as np


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: list
    n_trust_regions: int = 1
    length_min: float = 0.5**10
    length_max: float = 3.2
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    """TurboState class 
    """
    def __post_init__(self):
        if self.failure_tolerance == float("nan"):
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size,
                     float(self.dim) / self.batch_size]))
        self.failure_counter = list(np.zeros(self.n_trust_regions))
        self.success_counter = list(np.zeros(self.n_trust_regions))


def update_state(state, Y_next, update_failure=True):
    """Update a TurboState with the optimization running
    """
    for i in range(state.n_trust_regions):
        if max(Y_next[i, :, :]) > state.best_value:
            state.success_counter[i] += 1
            state.failure_counter[i] = 0
        else:
            state.success_counter[i] = 0
            if update_failure:
                state.failure_counter[i] += 1
        if state.success_counter[
                i] == state.success_tolerance:  # Expand trust region
            state.length[i] = min(2.0 * state.length[i], state.length_max)
            state.success_counter[i] = 0
        elif state.failure_counter[
                i] == state.failure_tolerance:  # Shrink trust region
            state.length[i] /= 2.0
            if update_failure:
                state.failure_counter[i] = 0

        state.best_value = max(state.best_value, max(Y_next.ravel()).item())
    if max(state.length) < state.length_min:
        state.length = state.length_min*16*np.ones(len(state.length))
        state.restart_triggered = True
    return state


def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        n_trust_regions=1,
        batch_size=1,
        X_pending=None,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        a = 0.2, 
        acqfn="ei"):
    """Generate a set of next best experimentes by TuRBO algorithm
    Parameters
    ----------
    state : TurboState
        Current state of TuRBO to determine the trust lengths
    model : GpyTorch Gaussian process model
        Surrogate model generated by GpyTorch GP interfaces
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
        Should be the same as the training X used in the GpyTorch GP model.
        Note TuRBO can only work for [0,1]^d, so the input needs to be normalized.
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
        Should be the same as the training Y used in the GpyTorch GP model.
    n_trust_regions: int, default=1
        Number of trust regions used in TuRBO. m value in TuRBO-m is the same as this
        n_trust_regions. Default is n_trust_regions=1 (TuRBO-1)
    batch_size : int, default=1
        Number of next experiments to be added
    X_pending : pyTorch tensor with a shape of (n_pending_samples, feature_size) of floats
        Current search space of experimental design after featurziation. 
    n_candidates : int, default=None
        Size of the sampling within the search space
    num_restarts : int, default=10
        Number of starting points for multistart acquisition function optimization.
    raw_samples : int, default=512
        Number of samples for initialization. 
    acqfn : str, default="ei"
        Acqusition function choices. Must be chosen from "ei", "pi", "ucb" and "ts"
    Returns
    -------
    X_next : pyTorch tensor with a shape of (batch_size, feature_size) of floats
        Selected experiments by TuRBO.
    acq_value : pyTorch tensor with a shape of of shape (batch_size,) of floats
        Acqusition values for the selected experiments
    Notes
    -----
    This implementation follows the tutorial from BoTorch 
    (https://botorch.org/tutorials/turbo_1) 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """

    dtype = X.dtype
    device = X.device
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    if model.covar_module.base_kernel.ard_num_dims == 1:
        weights = torch.tensor([1])
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    if acqfn == "ts":
        from botorch.generation import MaxPosteriorSampling
        if X_pending == None:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dim, scramble=True)
            pert = torch.zeros((n_trust_regions, n_candidates, dim))
            for t in range(n_trust_regions):
                tr_lb = torch.clamp(x_center - weights * state.length[t] / 2.0,
                                    0.0, 1.0)
                tr_ub = torch.clamp(x_center + weights * state.length[t] / 2.0,
                                    0.0, 1.0)
                pert[t, :, :] = tr_lb + (tr_ub - tr_lb) * sobol.draw(
                    n_candidates).to(dtype=dtype, device=device)

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (torch.rand(
                n_trust_regions, n_candidates, dim, dtype=dtype, device=device)
                    <= prob_perturb)
            ind = torch.where(mask.sum(dim=2) == 0)[0]
            mask[ind,
                 torch.randint(0, dim -
                               1, size=(len(ind), ), device=device)] = 1
            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_trust_regions, n_candidates,
                                     dim).clone()
            X_cand[mask] = pert[mask]
            assert X_cand.shape == (n_trust_regions, n_candidates, dim)

        else:
            if X_pending.shape[0] < n_candidates:
                n_candidates = int(0.8 * X_pending.shape[0])
            X_cand = torch.zeros((n_trust_regions, n_candidates, dim),
                                 dtype=dtype,
                                 device=device)
            for t in range(n_trust_regions):
                id_choice = np.random.choice(range(X_pending.shape[0]),
                                             n_candidates,
                                             replace=False)
                X_cand[t, :, :] = X_pending[id_choice, :].to(dtype=dtype,
                                                             device=device)

        # Sample on the candidate points
        X_next_m = torch.zeros((n_trust_regions, batch_size, dim),
                               dtype=dtype,
                               device=device)
        thompson_sampling = MaxPosteriorSampling(model=model,
                                                 replacement=False)
        for t in range(n_trust_regions):
            with torch.no_grad(): 
                X_next_m[t, :, :] = thompson_sampling(X_cand[t, :, :],
                                                  num_samples=batch_size)
        acq_value = None

    elif acqfn == "ei" or acqfn == 'pi' or acqfn == 'ucb':
        import botorch.acquisition as acqf
        from botorch.optim import optimize_acqf, optimize_acqf_discrete

        X_next_m = torch.zeros((n_trust_regions, batch_size, dim),
                               dtype=dtype,
                               device=device)
        acq_value = torch.zeros((n_trust_regions, batch_size),
                                dtype=dtype,
                                device=device)
        for t in range(n_trust_regions):
            if acqfn == "ei":
                # acq = acqf.monte_carlo.qExpectedImprovement(model,
                #                                             Y.max(),
                #                                             maximize=True)
                acq = acqf.analytic.ExpectedImprovement(model,
                                                            Y.max(),
                                                            maximize=True)                
            elif acqfn == "pi":
                acq = acqf.monte_carlo.qProbabilityOfImprovement(
                    model, Y.max())
            if acqfn == "ucb":
#                acq = acqf.monte_carlo.qUpperConfidenceBound(model, 0.2)
                acq = acqf.analytic.UpperConfidenceBound(model, a)
            tr_lb = torch.clamp(x_center - weights * state.length[t] / 2.0,
                                0.0, 1.0)
            tr_ub = torch.clamp(x_center + weights * state.length[t] / 2.0,
                                0.0, 1.0)
            if X_pending == None:
#                try:
                    X_next_m[t, :, :], acq_value[t, :] = optimize_acqf(
                        acq,
                        bounds=torch.stack([tr_lb, tr_ub]),
                        q=batch_size,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples)

                # except:
                #     print(tr_lb, tr_ub)
                #     X_next_m[t, :, :], acq_value[t, :] = torch.rand(
                #     batch_size).to(dtype=dtype, device=device), torch.zeros(batch_size)

            else:
                print('Dis')
                X_diff_ub, X_diff_lb = torch.max(
                    torch.sub(X_pending, tr_ub),
                    1)[0], torch.min(torch.sub(X_pending, tr_lb), 1)[0]
                index = np.where(np.logical_and(X_diff_ub <= 0,
                                                X_diff_lb >= 0))[0]
                if len(index) == 0:
                    index = np.arange(X_pending.shape[0])
                X_next_m[t, :, :], acq_value[t, :] = optimize_acqf_discrete(
                    acq,
                    choices=X_pending[index, :],
                    q=batch_size,
                    max_batch_size=2048)

    return X_next_m, acq_value

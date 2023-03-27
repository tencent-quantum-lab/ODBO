"""Gaussian process regressions"""

from __future__ import annotations
from typing import Any, Dict, Optional
import warnings
import torch
from torch import Tensor
from botorch import settings
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.utils.containers import TrainingData
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, StudentTLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.module import Module
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class GP(BatchedMultiOutputGPyTorchModel, ExactGP):
    """Exact single task Gaussian process regression.
    Notes
    -----
    This implementation modifies the source codes from BoTorch by allowing changes 
    of min_inferred_noise_level
    (https://botorch.org/api/_modules/botorch/models/gp_regression.html#SingleTaskGP) 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """

    def __init__(self,
                 train_X: Tensor,
                 train_Y: Tensor,
                 likelihood: Optional[Likelihood] = None,
                 covar_module: Optional[Module] = None,
                 outcome_transform: Optional[OutcomeTransform] = None,
                 input_transform: Optional[InputTransform] = None,
                 min_inferred_noise_level: Optional[Float] = 1e-4) -> None:
        """
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            min_inferred_noise_level: minimum value of added noises to kernel 
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        self._min_inferred_noise_level = min_inferred_noise_level

        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (
                noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    self._min_inferred_noise_level,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(self, train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        if train_X.shape[0] <  train_X.shape[1]:
            ard_num_dims = 1
        else:
            ard_num_dims = transformed_X.shape[-1] 
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        else:
            self.covar_module = covar_module
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData,
                         **kwargs: Any) -> Dict[str, Any]:
        """Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}


class HeteroskedasticGP(GP):
    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            train_Yvar: Tensor,
            likelihood: Optional[Likelihood] = None,
            covar_module: Optional[Module] = None,
            outcome_transform: Optional[OutcomeTransform] = None,
            input_transform: Optional[InputTransform] = None,
            min_inferred_noise_level: Optional[Float] = 1e-4) -> None:

        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
            self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
            validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
            self._set_dimensions(train_X=train_X, train_Y=train_Y)
            noise_likelihood = GaussianLikelihood(
                noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_inferred_noise_level, transform=None, initial_value=1.0
                ),
            )
            noise_model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Yvar,
                likelihood=noise_likelihood,
                outcome_transform=Log(),
                input_transform=input_transform,
            )
            if likelihood is None:
                likelihood = _GaussianLikelihoodBase(HeteroskedasticNoise(noise_model))
                super().__init__(
                    train_X=train_X,
                    train_Y=train_Y,
                    likelihood=likelihood,
                    input_transform=input_transform,
                )
                self.register_added_loss_term("noise_added_loss")
                self.update_added_loss_term(
                    "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
                )
            else:
                self._is_custom_likelihood = True
                
            if outcome_transform is not None:
                self.outcome_transform = outcome_transform
            self.to(train_X)

        def condition_on_observations(
            self, X: Tensor, Y: Tensor, **kwargs: Any
        ) -> HeteroskedasticSingleTaskGP:
            raise NotImplementedError

        def subset_output(self, idcs: List[int]) -> HeteroskedasticSingleTaskGP:
            raise NotImplementedError



class ApproGP(BatchedMultiOutputGPyTorchModel, ApproximateGP):

    def __init__(self,
                 train_X: Tensor,
                 train_Y: Tensor,
                 likelihood: Optional[Likelihood] = None,
                 covar_module: Optional[Module] = None,
                 outcome_transform: Optional[OutcomeTransform] = None,
                 input_transform: Optional[InputTransform] = None,
                 min_inferred_noise_level: Optional[Float] = 1e-4) -> None:
        """
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            min_inferred_noise_level: minimum value of added noises to kernel 
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        self._min_inferred_noise_level = min_inferred_noise_level

        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (
                noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    self._min_inferred_noise_level,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        variational_distribution = CholeskyVariationalDistribution(
            train_X.size(0))
        variational_strategy = VariationalStrategy(
            self,
            train_X,
            variational_distribution,
            learn_inducing_locations=False)
        ApproximateGP.__init__(self, variational_strategy)

        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        if train_X.shape[0] <  train_X.shape[1]:
            ard_num_dims = 1
        else:
            ard_num_dims = transformed_X.shape[-1]

        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        else:
            self.covar_module = covar_module
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData,
                         **kwargs: Any) -> Dict[str, Any]:
        """Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}


class StudentTGP(BatchedMultiOutputGPyTorchModel, ApproximateGP):
    """Single task Gaussian process regression with student T likelihood
    Notes
    -----
    This implementation refers to the source codes from BoTorch
    (https://botorch.org/api/_modules/botorch/models/gp_regression.html#SingleTaskGP) 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """

    def __init__(self,
                 train_X: Tensor,
                 train_Y: Tensor,
                 likelihood: Optional[Likelihood] = None,
                 covar_module: Optional[Module] = None,
                 outcome_transform: Optional[OutcomeTransform] = None,
                 input_transform: Optional[InputTransform] = None,
                 min_inferred_noise_level: Optional[Float] = 1e-4) -> None:
        """
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            min_inferred_noise_level: minimum value of added noises to kernel 
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        self._min_inferred_noise_level = min_inferred_noise_level
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (
                noise_prior.concentration - 1) / noise_prior.rate
            likelihood = StudentTLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    self._min_inferred_noise_level,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        variational_distribution = CholeskyVariationalDistribution(
            train_X.size(0))
        variational_strategy = VariationalStrategy(
            self,
            train_X,
            variational_distribution,
            learn_inducing_locations=False)
        ApproximateGP.__init__(self, variational_strategy)

        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        if train_X.shape[0] <  train_X.shape[1]:
            ard_num_dims = 1
        else:
            ard_num_dims = transformed_X.shape[-1]

        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        else:
            self.covar_module = covar_module
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData,
                         **kwargs: Any) -> Dict[str, Any]:
        """Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}

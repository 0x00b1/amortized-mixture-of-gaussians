import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Dirichlet
from typing import Dict, Tuple, Optional

class GaussianMixtureModelDataset(Dataset):
    """
    PyTorch Dataset that creates random Gaussian Mixture Models (GMMs) while maintaining
    a minimum distance between each component's mean vector. Supports custom device placement,
    optional random sampling of component weights (via a Dirichlet distribution), and separate
    methods for clarity and potential reuse.

    Args:
        size (int): Number of GMMs in the dataset. Default: 1000
        n (int): Total samples drawn from each GMM. Default: 100
        minimum_k (int): Minimum number of mixture components. Default: 1
        maximum_k (int): Maximum number of mixture components. Default: 5
        dimension (int): Dimensionality of each component. Default: 2
        minimum_distance (float): Minimum distance between any two means. Default: 1.0
        minimum_log_variance (float): Minimum log variance per dimension. Default: -1.0
        maximum_log_variance (float): Maximum log variance per dimension. Default: 1.0
        max_attempts (int): Maximum tries to place each mean before raising an error. Default: 1000
        random_scale (float): Scaling factor for randomly drawn means. Default: 5.0
        use_random_weights (bool): If True, sample weights from a Dirichlet distribution
            rather than splitting samples uniformly among components. Default: False
        seed (Optional[int]): Optional random seed for reproducibility. Default: None
        device (torch.device or str or None): PyTorch device (e.g., "cpu" or "cuda"). Default: None
    """

    def __init__(
        self,
        size: int = 1000,
        n: int = 100,
        minimum_k: int = 1,
        maximum_k: int = 5,
        dimension: int = 2,
        minimum_distance: float = 1.0,
        minimum_log_variance: float = -1.0,
        maximum_log_variance: float = 1.0,
        max_attempts: int = 1000,
        random_scale: float = 5.0,
        use_random_weights: bool = False,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        self.size = size
        self.n = n
        self.minimum_k = minimum_k
        self.maximum_k = maximum_k
        self.dimension = dimension
        self.minimum_distance = minimum_distance

        self.minimum_log_variance = minimum_log_variance
        self.maximum_log_variance = maximum_log_variance

        self.log_variance_difference = self.maximum_log_variance - self.minimum_log_variance

        self.max_attempts = max_attempts
        self.random_scale = random_scale
        self.use_random_weights = use_random_weights

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(
            self,
            index: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        k = random.randint(self.minimum_k, self.maximum_k + 1)

        components = Tensor(
            k,
            device=self.device,
        )

        means = torch.zeros(
            self.maximum_k,
            self.dimension,
            device=self.device,
        )

        log_variances = torch.zeros(
            self.maximum_k,
            self.dimension,
            device=self.device,
        )

        masks = torch.zeros(
            self.maximum_k,
            device=self.device,
        )

        self._sample_means(k, means, log_variances, masks)

        if self.use_random_weights:
            distribution = Dirichlet(
                concentration=torch.ones(
                    k,
                    device=self.device,
                ),
            )

            ks = torch.floor(distribution.sample() * self.n)

            for index in range(self.n - torch.sum(ks)):
                ks[index % k] = ks[index % k] + 1
        else:
            ks = torch.full(
                [k],
                self.n // k,
                dtype=torch.int64,
                device=self.device,
            )

            ks[-1] = ks[-1] + self.n % k

        samples = []

        for j in range(k):
            covariance_matrix = torch.diag(torch.exp(log_variances[j]))

            distribution = MultivariateNormal(
                means[j],
                covariance_matrix=covariance_matrix,
            )

            sample = distribution.sample([ks[j]])

            samples = [*samples, sample]

        return (
            torch.concatenate(samples, dim=0),
            (
                components,
                means,
                log_variances,
                masks,
            )
        )

    def _sample_means(
        self,
        k: int,
        means: Tensor,
        log_variances: Tensor,
        masks: Tensor,
    ) -> None:
        chosen_means = []

        for j in range(k):
            for _ in range(self.max_attempts):
                candidate_mean = torch.randn(self.dimension, device=self.device)

                candidate_mean = candidate_mean * self.random_scale

                candidates = []

                for chosen_mean in chosen_means:
                    candidate = torch.norm(candidate_mean - chosen_mean) >= self.minimum_distance

                    candidates = [*candidates, candidate]

                if all(candidates):
                    chosen_means.append(candidate_mean)

                    means[j] = candidate_mean

                    log_variance = torch.rand(
                        self.dimension,
                        device=self.device,
                    )

                    log_variances[j] = log_variance * self.log_variance_difference + self.minimum_log_variance

                    masks[j] = 1

                    break
            else:
                raise RuntimeError

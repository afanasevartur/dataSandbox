import numpy as np
from typing import List, Union, Optional

class CoreGenerator:
    """
    testing kit stage 1
    """
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def generate_integers(self, low: int, high: int, size: int) -> np.ndarray:
        return self.rng.integers(low, high, size=size)

    def generate_floats(self, low: float, high: float, size: int) -> np.ndarray:
        return self.rng.uniform(low, high, size=size)

    def generate_normal_distribution(self, mean: float, std_dev: float, size: int) -> np.ndarray:
        return self.rng.normal(loc=mean, scale=std_dev, size=size)

    def generate_choices(self, elements: List[Union[str, int]], size: int, probabilities: Optional[List[float]] = None) -> np.ndarray:
        """
        test 1 
        """
        return self.rng.choice(elements, size=size, p=probabilities)


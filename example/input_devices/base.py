from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class InputDeviceBase(ABC):
    @abstractmethod
    def get_fingers_data(self) -> Dict[str, np.ndarray]:
        """Return a dict with `left_fingers` and `right_fingers` data."""
        pass

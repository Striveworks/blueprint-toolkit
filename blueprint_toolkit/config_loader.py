"""An interface and local implementation for loading training run configurations."""

import json
from abc import ABC, abstractmethod
from typing import Any


class ConfigLoader(ABC):
    """Interface for a class that loads a run's config."""

    @abstractmethod
    def load_config(self) -> dict[Any, Any]:
        """Load the run configuration.

        Returns
        -------
        dict[Any, Any]
            Config

        """
        raise NotImplementedError()


class FileConfigLoader(ConfigLoader):
    """Loads config from a json file.

    Parameters
    ----------
    config_path : str
        Local path to the JSON file containing configuration data.

    """

    config_path: str
    """Local path to the file that holds configuration"""

    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_config(self) -> dict[Any, Any]:
        """Load and return the configuration from the JSON file.

        Returns
        -------
        dict[Any, Any]
            The configuration data loaded from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified `config_path` does not exist.
        json.JSONDecodeError
            If the JSON file is invalid and cannot be decoded.

        """
        with open(self.config_path) as f:
            return json.load(f)


class MemoryConfigLoader(ConfigLoader):
    """Returns specified config.

    Parameters
    ----------
    config : dict[Any, Any]
        The configuration data to be returned by ``load_config()``.

    """

    config: dict[Any, Any]
    """Configuration to return"""

    def __init__(self, config: dict[Any, Any]):
        self.config = config

    def load_config(self) -> dict[Any, Any]:
        """Return the pre-specified configuration data.

        Returns
        -------
        dict[Any, Any]
            The configuration data provided during initialization.

        Notes
        -----
        This method simply returns the `config` attribute set during initialization.

        """
        return self.config

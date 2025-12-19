"""An interface and local implementation for loading models."""

from abc import ABC
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ModelNotFoundError


@dataclass
class LoadModelData:
    """Loaded model details."""

    dir: Path
    """Path to a directory that contains the model's files"""
    id: str
    """Unique identifier for the model"""


class ModelLoader(ABC):
    """Interface for model loaders."""

    @contextmanager
    def load_model(self, id: str) -> Generator[LoadModelData, None, None]:
        """Load a model.

        Acts as a context manager which returns an object containing relevant details
        about the model.

        Parameters
        ----------
        id : str
            Unique identifier for model.

        Yields
        ------
        LoadModelData
            Details about the loaded model.

        Raises
        ------
        ModelNotFoundError
            If the model with the specified id or the most recent model
            cannot be found.

        Example
        -------
        .. code-block:: python

            with model_loader.load_model(id) as model:
                model.load_state_dict(torch.load(model.dir / "model.pth"))

        """
        raise NotImplementedError()


class LocalFileModelLoader(ModelLoader):
    """Manages models in a local directory.

    Parameters
    ----------
    base_dir : str | Path
        The base directory where models will be loaded from.
        Individual models are loaded from the directory path
        ``{base_dir}/models/{model_id}/``.

    """

    base_dir: Path
    """Local directory where models will be stored"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    @contextmanager
    def load_model(self, id: str) -> Generator[LoadModelData, None, None]:
        """Load a model.

        Acts as a context manager which returns an object containing relevant details
        about the model.

        Parameters
        ----------
        id : str
            Unique identifier for model.

        Yields
        ------
        LoadModelData
            Details about the loaded model.

        Raises
        ------
        ModelNotFoundError
            If the model with the specified id or the most recent model
            cannot be found.

        Example
        -------
        .. code-block:: python

            with model_loader.load_model(id) as model:
               model.load_state_dict(torch.load(model.dir / "model.pth"))

        """
        model_path = self._model_path(id=id)
        if not model_path.exists():
            raise ModelNotFoundError(id=id)
        yield LoadModelData(dir=model_path, id=id)

    def _models_path(self) -> Path:
        return self.base_dir / "models"

    def _model_path(self, id: str) -> Path:
        return self._models_path() / id

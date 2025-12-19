"""An interface and local implementation for saving and loading training run checkpoints."""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .exceptions import CheckpointNotFoundError

log = logging.getLogger(__name__)


@dataclass
class SaveCheckpointData:
    """Information to facilitate checkpoint saving."""

    dir: Path
    """Directory path where checkpoint files should be saved"""


@dataclass
class LoadCheckpointData:
    """Loaded checkpoint details."""

    dir: Path
    """Path to a directory that contains the checkpoint's files"""
    id: str
    """Unique identifier of the checkpoint"""
    global_step: int
    """Global step associated with the checkpoint"""
    run_id: str
    """Unique identifier of the run associated with this checkpoint"""


class CheckpointManager(ABC):
    """Interface for checkpoint managers."""

    @abstractmethod
    @contextmanager
    def save_checkpoint(
        self, global_step: int
    ) -> Generator[SaveCheckpointData, None, None]:
        """Save a checkpoint.

        Acts as a context manager which returns an object that includes the path to
        the directory where the checkpoint's files should be saved. Once the context is
        exited, all files and directories in the returned path will be saved in a
        checkpoint at the specified global step.

        Parameters
        ----------
        global_step : int
            Global training step to save a checkpoint at.

        Yields
        ------
        SaveCheckpointData
            Information to facilitate checkpoint saving.

        Example
        -------
        .. code-block:: python

            with checkpoint_manager.save_checkpoint(global_step=1) as checkpoint:
                torch.save(model.state_dict(), checkpoint.dir / "model.pth")

        """
        raise NotImplementedError()

    @abstractmethod
    @contextmanager
    def load_checkpoint(
        self,
        id: str | None = None,
    ) -> Generator[LoadCheckpointData, None, None]:
        """Load a checkpoint.

        Acts as a context manager which returns an object with information about the
        loaded checkpoint such as a path to the directory containing the checkpoint's
        files.

        Parameters
        ----------
        id : str | None, optional
            Unique identifier for checkpoint.
            If no id is specified, the most recent checkpoint from the current run will be loaded.

        Yields
        ------
        LoadCheckpointData
            Details about the loaded checkpoint.

        Raises
        ------
        CheckpointNotFoundError
            If the checkpoint with the specified id or the most recent checkpoint
            cannot be found.

        Example
        -------
        .. code-block:: python

            with checkpoint_manager.load_checkpoint(id) as checkpoint:
                model.load_state_dict(torch.load(checkpoint.dir / "model.pth"))

        """
        raise NotImplementedError()


class LocalFileCheckpointManager(CheckpointManager):
    """Manages checkpoints in a local directory.

    Parameters
    ----------
    run_id : str
        The unique identifier for the run. This will be used to construct a
        unique path for stored files.
    base_dir : str
        The base directory where checkpoints will be stored.
        Individual checkpoints are stored under the directory path
        ``{base_dir}/runs/{run_id}/checkpoints/{checkpoint_id}/``.

    """

    run_id: str
    """Unique identifier used to construct a unique path to stored files"""
    base_dir: Path
    """Local directory where checkpoints will be stored"""

    def __init__(self, run_id: str, base_dir: str | Path):
        self.run_id = run_id
        self.base_dir = Path(base_dir)

    @contextmanager
    def save_checkpoint(
        self, global_step: int
    ) -> Generator[SaveCheckpointData, None, None]:
        """Save a checkpoint.

        Acts as a context manager which returns an object that includes the path to
        the directory where the checkpoint's files should be saved. Once the context is
        exited, all files and directories in the returned path will be saved in a
        checkpoint at the specified global step. If an exception is raised during the
        context, the directory and all its contents will be deleted.

        Parameters
        ----------
        global_step : int
            Global training step to save a checkpoint at.

        Yields
        ------
        SaveCheckpointData
            Information to facilitate checkpoint saving.

        Example
        -------
        .. code-block:: python

            with checkpoint_manager.save_checkpoint(global_step=1) as checkpoint:
                torch.save(model.state_dict(), checkpoint.dir / "model.pth")

        """
        checkpoint_id = f"{global_step}-{self.run_id}__{str(uuid4())}"
        self._most_recent_checkpoint_id = checkpoint_id
        checkpoint_path = self._checkpoint_path(run_id=self.run_id, id=checkpoint_id)
        os.makedirs(checkpoint_path, exist_ok=True)

        try:
            yield SaveCheckpointData(dir=checkpoint_path)
        except Exception as e:
            shutil.rmtree(checkpoint_path)
            raise e

    @contextmanager
    def load_checkpoint(
        self,
        id: str | None = None,
    ) -> Generator[LoadCheckpointData, None, None]:
        """Load a checkpoint.

        Acts as a context manager which returns an object with information about the
        loaded checkpoint such as a path to the directory containing the checkpoint's
        files.

        Parameters
        ----------
        id : str | None, optional
            Unique identifier for checkpoint.
            If no id is specified, the most recent checkpoint from the current run will be loaded.

        Yields
        ------
        LoadCheckpointData
            Details about the loaded checkpoint

        Raises
        ------
        CheckpointNotFoundError
            If the checkpoint with the specified id or the most recent checkpoint
            cannot be found.

        Example
        -------
        .. code-block:: python

            with checkpoint_manager.load_checkpoint(id) as checkpoint:
                global_step = checkpoint.global_step
                model.load_state_dict(torch.load(checkpoint.dir / "model.pth"))

        """
        global_step = None
        if id is None:
            checkpoints_path = self._checkpoints_path(run_id=self.run_id)
            checkpoint_paths = list(checkpoints_path.glob("*"))
            if len(checkpoint_paths) == 0:
                raise CheckpointNotFoundError()
            checkpoint_path = max(
                checkpoint_paths,
                key=lambda c: (c.stat().st_mtime, int(c.name.split("-")[0])),
            )
        else:
            runs_path = self._runs_path()
            checkpoint_paths = list(runs_path.glob(self._checkpoint_glob(id=id)))
            if len(checkpoint_paths) == 0:
                raise CheckpointNotFoundError(id=id)
            if len(checkpoint_paths) > 1:
                raise RuntimeError(f"unexpected multiple checkpoints with id {id!r}")
            checkpoint_path = checkpoint_paths[0]

        id = checkpoint_path.name
        base, _ = id.rsplit("__", 1)
        parts = base.split("-")
        global_step = int(parts[0])
        run_id = "-".join(parts[1:])
        yield LoadCheckpointData(
            dir=checkpoint_path, id=id, global_step=global_step, run_id=run_id
        )

    def _runs_path(self) -> Path:
        return self.base_dir / "runs"

    def _checkpoints_path(self, *, run_id: str) -> Path:
        return self._runs_path() / run_id / "checkpoints"

    def _checkpoint_glob(self, *, id: str) -> str:
        return f"*/checkpoints/{id}"

    def _checkpoint_path(self, *, run_id: str, id: str) -> Path:
        return self._checkpoints_path(run_id=run_id) / id

"""Module for managing the local execution context of a training run.

This module provides classes and utilities for managing various aspects
of a training run, including configuration loading, dataset fetching, model loading,
progress saving, metric saving, and checkpoint loading and saving.
"""

import logging
import signal
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any

from .checkpoint_manager import (
    CheckpointManager,
    LoadCheckpointData,
    LocalFileCheckpointManager,
    SaveCheckpointData,
)
from .config_loader import (
    ConfigLoader,
    MemoryConfigLoader,
)
from .dataset import (
    DatasetFetcher,
    LocalFileDatasetFetcher,
)
from .exceptions import (
    RunContextInterruptedError,
)
from .metric_saver import (
    MemoryMetricSaver,
    MetricSaver,
    SaveMetricDict,
)
from .model_loader import (
    LoadModelData,
    LocalFileModelLoader,
    ModelLoader,
)
from .progress_saver import (
    MemoryProgressSaver,
    ProgressSaver,
    SaveProgressDict,
)

logger = logging.getLogger()


def _is_context_manager(obj):
    return hasattr(obj, "__enter__") and hasattr(obj, "__exit__")


class RunContext:
    """Class for managing interactions with a training run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the training run.
    config_loader : ConfigLoader
        The ``ConfigLoader`` implementation to use.
    dataset_fetcher : DatasetFetcher
        The ``DatasetFetcher`` implementation to use.
    model_loader : ModelLoader
        The ``ModelLoader`` implementation to use.
    progress_saver : ProgressSaver
        The ``ProgressSaver`` implementation to use.
    metric_saver : MetricSaver
        The ``MetricSaver`` implementation to use.
    checkpoint_manager : CheckpointManager
        The ``CheckpointManager`` implementation to use.

    """

    run_id: str
    """Unique ID to track the run being managed"""
    config_loader: ConfigLoader
    """Method for loading run configuration"""
    dataset_fetcher: DatasetFetcher
    """Class for fetching datums"""
    model_loader: ModelLoader
    """Loads models"""
    progress_saver: ProgressSaver
    """Saves progress"""
    metric_saver: MetricSaver
    """Saves metrics"""
    checkpoint_manager: CheckpointManager
    """Manages checkpoints"""
    _exit_stack: ExitStack
    """Exit stack"""

    def __init__(
        self,
        *,
        run_id: str,
        config_loader: ConfigLoader,
        dataset_fetcher: DatasetFetcher,
        model_loader: ModelLoader,
        progress_saver: ProgressSaver,
        metric_saver: MetricSaver,
        checkpoint_manager: CheckpointManager,
    ):
        def handle_interruption(signum, frame):
            raise RunContextInterruptedError()

        signal.signal(signal.SIGINT, handle_interruption)
        signal.signal(signal.SIGTERM, handle_interruption)

        self.run_id = run_id
        self.config_loader = config_loader
        self.dataset_fetcher = dataset_fetcher
        self.model_loader = model_loader
        self.progress_saver = progress_saver
        self.metric_saver = metric_saver
        self.checkpoint_manager = checkpoint_manager
        self._exit_stack = ExitStack()

    def __enter__(self):
        """Enter method for context management.

        Enters any context managers associated with the ``config_loader``, ``model_loader``,
        ``progress_writer``, ``metric_writer``, and ``checkpoint_manager`` into the exit stack.

        Returns
        -------
        RunContext
            The current instance of the RunContext with entered context managers.

        """
        self._exit_stack.__enter__()
        if _is_context_manager(self.config_loader):
            self._exit_stack.enter_context(self.config_loader)  # type: ignore[arg-type]
        if _is_context_manager(self.model_loader):
            self._exit_stack.enter_context(self.model_loader)  # type: ignore[arg-type]
        if _is_context_manager(self.progress_saver):
            self._exit_stack.enter_context(self.progress_saver)  # type: ignore[arg-type]
        if _is_context_manager(self.metric_saver):
            self._exit_stack.enter_context(self.metric_saver)  # type: ignore[arg-type]
        if _is_context_manager(self.checkpoint_manager):
            self._exit_stack.enter_context(self.checkpoint_manager)  # type: ignore[arg-type]
        return self

    def __exit__(
        self, exception_type, exception_value, exception_traceback
    ) -> bool | None:
        """Exit method for context management.

        Exits the context stack, ensuring all context managers are properly exited.

        Parameters
        ----------
        exception_type : type
            Type of the exception raised (or None if no exception).
        exception_value : Exception
            The exception instance raised (or None if no exception).
        exception_traceback : traceback
            Traceback object associated with the exception (or None if no exception).


        Returns
        -------
        bool | None
            ``True`` if a ``RunContextInterruptedError`` is raised, which will
            suppress the exception.

        """
        exit_stack_exit = self._exit_stack.__exit__(
            exception_type, exception_value, exception_traceback
        )
        if isinstance(exception_value, RunContextInterruptedError):
            return True
        return exit_stack_exit

    def load_config(self) -> dict[Any, Any]:
        """Load a training run configuration.

        Returns
        -------
        dict[Any, Any]
            The training run configuration.

        Examples
        --------
        .. code-block:: python

            with run_context:
                run_config = run_context.load_config()
                config = Config.model_validate(run_config)

        """
        return self.config_loader.load_config()

    @contextmanager
    def load_model(self, id: str) -> Generator[LoadModelData, None, None]:
        """Load a model.

        Acts as a context manager which returns an object containing relevant details
        about the model.

        Parameters
        ----------
        id : str
            Unique identifier for the model.

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
        with self.model_loader.load_model(id=id) as model:
            yield model

    def save_metrics(self, metrics: list[SaveMetricDict]):
        """Save training metrics.

        Parameters
        ----------
        metrics : list[Metric]
            Metrics to save.

        Examples
        --------
        .. code-block:: python

            run_context.save_metrics(
                metrics=[
                    SaveMetricDict(
                        global_step=global_step,
                        tag="train/loss",
                        value=loss,
                    )
                ],
           )

        """
        self.metric_saver.save_metrics(metrics=metrics)

    def save_progress(
        self,
        progress: list[SaveProgressDict],
    ):
        """Upload progresses.

        Parameters
        ----------
        progress : list[SaveProgressDict]
            The current progress to save.

        Examples
        --------
        .. code-block:: python

            run_context.save_progress(
                [
                    SaveProgressDict(
                        operation="Training",
                        value=global_step,
                        final_value=n_global_steps,
                        units="steps",
                    ),
                    SaveProgressDict(
                        operation="Evaluating",
                        value=eval_step,
                        final_value=total_eval_steps,
                        units="steps",
                    ),
                ],
           )

        """
        self.progress_saver.save_progress(progress=progress)

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
            Global training step to save the checkpoint at.

        Yields
        ------
        SaveCheckpointData
            Information to facilitate checkpoint saving.

        Examples
        --------
        .. code-block:: python

            with run_context.save_checkpoint(global_step=1) as checkpoint:
                model_path = checkpoint.dir / MODEL_FILE_NAME
                torch.save(model.state_dict(), model_path)
                optimizer_path = checkpoint.dir / OPTIMIZER_FILE_NAME)
                torch.save(optimizer.state_dict(), optimizer_path)

        """
        with self.checkpoint_manager.save_checkpoint(
            global_step=global_step
        ) as checkpoint:
            yield checkpoint

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
        id : str | None
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

            with run_context.load_checkpoint(id) as checkpoint:
                global_step = checkpoint.global_step
                model.load_state_dict(torch.load(checkpoint.dir / "model.pth"))

        """
        with self.checkpoint_manager.load_checkpoint(id=id) as checkpoint:
            yield checkpoint


def local_run_context(
    *,
    run_id: str,
    config: dict[Any, Any],
    base_dir: str,
    dataset_fetcher: DatasetFetcher | None = None,
) -> RunContext:
    """Return a RunContext manager.

    Intended to be used locally for development and testing.

    Example
    -------
    .. code-block:: python

        with local_run_context(
            run_id=run_id,
            config={},
            base_dir="./test"
        ) as run_context:
            tag = "metric" + str(time.time())
            metrics = [dict(global_step=1, tag=tag, value=1.0)]
            run_context.save_metrics(metrics=metrics)

    Parameters
    ----------
    run_id : int
        Unique identifier for the training run.

    config : dict[Any, Any]
        The training run configuration to be used.

    base_dir : str
        Local directory where checkpoints will be stored.

    dataset_fetcher: DatasetFetcher | None
        dataset fetcher to retrieve datums and metadata

    Returns
    -------
    RunContext
        Manager for local interactions suitable for testing and development.

    """
    return RunContext(
        run_id=run_id,
        config_loader=MemoryConfigLoader(config=config),
        dataset_fetcher=dataset_fetcher or LocalFileDatasetFetcher(base_dir=base_dir),
        model_loader=LocalFileModelLoader(base_dir=base_dir),
        progress_saver=MemoryProgressSaver(),
        metric_saver=MemoryMetricSaver(),
        checkpoint_manager=LocalFileCheckpointManager(run_id=run_id, base_dir=base_dir),
    )

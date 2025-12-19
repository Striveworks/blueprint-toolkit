"""An interface and local implementation for saving training run progress."""

from abc import ABC, abstractmethod
from typing import TypedDict


class SaveProgressDict(TypedDict):
    """Training run progress."""

    operation: str
    """The name of the task that progress is being reported on (e.g. "Evaluation")"""
    value: float
    """Current value of the operation's progress"""
    final_value: float
    """The expected final value of the operation's progress"""
    units: str
    """Units that the progress is measured in"""


class ProgressSaver(ABC):
    """Interface for class that saves progress."""

    @abstractmethod
    def save_progress(self, progress: list[SaveProgressDict]):
        """Save training run progress.

        Parameters
        ----------
        progress : list[SaveProgressDict]
            The current progress to save.

        """
        raise NotImplementedError()


class MemoryProgressSaver(ProgressSaver):
    """Saves progress in memory to this object."""

    current_progress: list[SaveProgressDict]
    """Current training run progress"""

    def save_progress(self, progress: list[SaveProgressDict]):
        """Save the training run progress in memory.

        Parameters
        ----------
        progress : list[SaveProgressDict]
            List of dictionaries representing progress to be saved.

        """
        self.current_progress = progress

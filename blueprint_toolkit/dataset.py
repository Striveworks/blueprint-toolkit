"""Module for interacting with datasets.

This module provides a framework-agnostic interface for fetching datasets.
"""

import abc
import json
from pathlib import Path
from typing import Any, TypedDict

from typing_extensions import NotRequired


class DatumMetadataDict(TypedDict):
    """Minimal metadata interface for a datum.

    This is a generic interface that implementations can extend.
    The actual implementation may return additional fields.
    """

    id: str
    """Unique identifier for the datum"""
    created_at: NotRequired[str]
    """Timestamp when the datum was created"""
    metadata: NotRequired[dict[str, Any]]
    """Additional metadata"""


class SnapshotDict(TypedDict):
    """A dataset snapshot identifier."""

    snapshot_id: str
    split: NotRequired[str | None]


class DatasetFetcher(abc.ABC):
    """Framework-agnostic helper interface for fetching datums."""

    def prepare_snapshot_split(
        self,
        snapshot_id: str,
        split: str | None,
    ):
        """Set up dataset fetcher for the snapshot id and split.

        Parameters
        ----------
        snapshot_id: str
            dataset snapshot identifier
        split: NotRequired[str | None]
            dataset split identifier

        """
        raise NotImplementedError()

    def get_datum_at_index(self, index: int) -> tuple[bytes, Any]:
        """Fetch datum information.

        Returns a datum and its metadata at a given index. Utilizes the datasets service default sort order.

        Parameters
        ----------
        index : int
            The index into the dataset to return

        Returns
        -------
        tuple[bytes, DatumDict]
            The datum downloaded as ``bytes`` and its associated metadata

        Raises
        ------
        ValueError
            If `index` is out of bounds for the dataset length
        RuntimeError
            If the datasets service returns an unexpected response

        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def snapshot(self) -> SnapshotDict:
        """The ``SnapshotDict`` for this dataset."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_class_labels(
        self,
    ) -> list[str]:
        """Get the sorted class labels in this snapshot split.

        Returns
        -------
        list[str]
            The sorted list of class labels

        Examples
        --------
        .. code-block:: python

            ["cat", "dog"]

        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_datums(self) -> int:
        """The number of datums in this snapshot split."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def task_type(self) -> str:
        """The task type of this snapshot split."""
        raise NotImplementedError()


class LocalFileDatasetFetcher(DatasetFetcher):
    """Framework-agnostic helper class for loading datums.

    Loads datums from the local files

    Parameters
    ----------
    snapshot_id : str
        The ID of the snapshot to load datums from.
    split : str | None
        The dataset split to load (e.g., "train", "val", "test").
        If None, defaults to loading all splits.
    file_root : str | Path, optional
        Root directory of the datums.

    Raises
    ------
    ValueError
        If the task type cannot be determined from the snapshot datum stats.

    """

    snapshot_id: str
    split: str | None
    file_root: Path

    def __init__(
        self,
        *,
        base_dir: str | Path = ".",
    ):
        self.file_root = Path(base_dir).absolute() / "datasets"
        self.file_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_id = ""
        self.split = None

    def prepare_snapshot_split(
        self,
        snapshot_id: str,
        split: str | None,
    ):
        """Set up dataset fetcher for the snapshot id and split.

        Parameters
        ----------
        snapshot_id: str
            dataset snapshot identifier
        split: str | None
            dataset split identifier

        """
        if self.snapshot_id == snapshot_id and self.split == split:
            return

        self.snapshot_id = snapshot_id
        self.split = split

        file_root = self.file_root / snapshot_id / str(split)
        file_root.mkdir(parents=True, exist_ok=True)
        self._cached_datum_index_to_id = {
            i: f.stem for i, f in enumerate(file_root.glob("*.json")) if f.is_file()
        }
        self._num_datums = len(self._cached_datum_index_to_id)

    def get_datum_at_index(self, index: int) -> tuple[bytes, Any]:
        """Fetch datum information.

        Returns a datum and its metadata at a given index. Utilizes the default sort order for file name.

        Parameters
        ----------
        index : int
            The index into the dataset to return

        Returns
        -------
        tuple[bytes, DatumDict]
            The datum downloaded as ``bytes`` and its associated metadata

        Raises
        ------
        ValueError
            If `index` is out of bounds for the dataset length
        RuntimeError
            If the datasets service returns an unexpected response

        """
        file_root = self.file_root / self.snapshot_id / str(self.split)

        if index < 0 or index >= self.num_datums:
            raise ValueError(
                f"index must be a non-negative integer less than "
                f"{self.num_datums}. given {index}"
            )
        datum_id = self._cached_datum_index_to_id.get(index)
        datum_bytes = (file_root / str(datum_id)).read_bytes()
        datum_dict = json.loads((file_root / (str(datum_id) + ".json")).read_text())
        return datum_bytes, datum_dict

    @property
    def snapshot(self) -> SnapshotDict:
        """The ``SnapshotDict`` for this dataset."""
        return SnapshotDict(snapshot_id=self.snapshot_id, split=self.split)

    def get_class_labels(
        self,
    ) -> list[str]:
        """Get the sorted class labels in this snapshot split.

        Returns
        -------
        list[str]
            The sorted list of class labels

        Examples
        --------
        .. code-block:: python

            ["cat", "dog"]

        """
        return []

    @property
    def num_datums(self) -> int:
        """The number of datums in this snapshot split."""
        return self._num_datums

    @property
    def task_type(self) -> str:
        """The task type of this snapshot id."""
        return ""

"""Config types for use in blueprint configurations."""

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from typing_extensions import NotRequired

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


@dataclass
class DatasetSnapshot:
    """A config type that specifies a dataset snapshot."""

    snapshot_id: str
    """ID of the Chariot dataset snapshot"""
    split: str | None = None  # train, val, or test
    """Split of the Chariot dataset snapshot"""


@dataclass
class Model:
    """A config type that specifies a model."""

    model_id: str
    """ID of the Chariot model"""


@dataclass
class Checkpoint:
    """A config type that specifies a training run checkpoint."""

    checkpoint_id: str
    """ID of the Chariot checkpoint"""


class ModelConfigDict(TypedDict):
    """Schema for cataloging a checkpoint as a model."""

    artifact_type: str
    """The artifact type or framework of the model.
    Examples: ``pytorch``, ``huggingface``, ``custom-engine``"""
    class_labels: NotRequired[dict[str, int]]
    """Mapping from class labels to integer IDs."""
    copy_key_suffixes: list[str]
    """List of S3 key suffixes from the checkpoint to include in the model."""

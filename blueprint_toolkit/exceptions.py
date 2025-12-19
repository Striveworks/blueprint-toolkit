"""Module defining custom exceptions for specific errors."""

# TODO fix


class ModelNotFoundError(Exception):
    """Exception for when a model cannot be found.

    Parameters
    ----------
    id : str
        The ID of the model that cannot be found.

    """

    def __init__(self, id: str):
        super().__init__(f"could not find a model with id {id}")


class CheckpointNotFoundError(Exception):
    """Exception for when a checkpoint cannot be found.

    Parameters
    ----------
    id : str | None, optional
        The ID of the checkpoint that cannot be found.

    """

    def __init__(self, id: str | None = None):
        if id:
            msg = f"could not find a checkpoint with id {id}"
        else:
            msg = "could not find the most recent checkpoint"
        super().__init__(msg)


class DatumNotFoundError(Exception):
    """Exception for when a datum cannot be found.

    Parameters
    ----------
    index : int
        The index of the datum that cannot be found.
    additional_info : str, optional
        Additional information about the error, defaults to None.

    """

    def __init__(self, index: int, additional_info: str | None = None) -> None:
        message = f"datum not found at index {index}"
        if additional_info:
            message += f". {additional_info}"
        super().__init__(message)


class RunContextInterruptedError(Exception):
    """Exception raised when handling termination signals."""

    def __init__(self) -> None:
        super().__init__("RunContext was interrupted")

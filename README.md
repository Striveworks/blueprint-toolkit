# Blueprint Toolkit

A Blueprint Toolkit is a Python toolkit for developing and running Training Blueprints locally. This library provides abstractions and utilities for managing Training Runs, including configuration loading, model management, checkpointing, metrics tracking, and dataset handling.

## Features

- **RunContext:** Manages the execution context of a Training Run with signal handling
- **ConfigLoader:** Loads and manages training configuration from files or memory
- **ModelLoader:** Loads models from local filesystem
- **CheckpointManager:** Saves and loads training checkpoints
- **ProgressSaver:** Tracks and saves training progress
- **MetricSaver:** Records and saves training metrics
- **Dataset Utilities:** Fetches and manages dataset Snapshots
- **S3 Utilities:** Uploads and downloads directories to/from S3

## Components

### RunContext

The `RunContext` class provides a unified interface for managing all aspects of a Training Run. It handles signal interruption and provides methods for loading configuration and models, as well as for managing checkpoints, metrics, and progress.

### ConfigLoader

Abstract interface for loading configuration. Implementations include:
- `FileConfigLoader`: Loads configuration from `yaml`/`json` files
- `MemoryConfigLoader`: Loads configuration from in-memory dictionaries

### ModelLoader

Abstract interface for loading models. Implementations include:
- `LocalFileModelLoader`: Loads models from local filesystem

### CheckpointManager

Abstract interface for managing checkpoints. Implementations include:
- `LocalFileCheckpointManager`: Manages checkpoints on local filesystem

### ProgressSaver and MetricSaver

Abstract interfaces for saving training progress and metrics. Implementations include:
- `MemoryProgressSaver`: Stores progress in memory
- `MemoryMetricSaver`: Stores metrics in memory

## Example: Create a local run context
```python
with local_run_context(
    run_id="my-training-run",
    config_path="config.yaml",
    model_path="model.pth",
    checkpoint_dir="checkpoints",
) as ctx:
    # Load configuration
    config = ctx.load_config()
    
    # Load model
    model_data = ctx.load_model()

    # your training loop implementation
    train(...)
    
    # Save metrics
    ctx.save_metric(name="loss", value=0.5, step=1)
    
    # Save progress
    ctx.save_progress(epoch=1, step=100)
    
    # Save checkpoint
    ctx.save_checkpoint(path="checkpoint.pth", metadata={"epoch": 1})
```

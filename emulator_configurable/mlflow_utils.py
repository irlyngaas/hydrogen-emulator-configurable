"""
MLflow-specific helpers, split out of utils.py so that importing utils.py
(or anything that imports it, like model_builder.py) doesn't force mlflow
to be installed. Only main.py's predict_subsurface (inference) actually
uses this module -- training no longer depends on MLflow at all.
"""
import os
import json
import tempfile
import mlflow
from glob import glob


def load_mlflow_credentials(mlflow_credentials_file):
    """
    Loads MLflow credentials from a file and sets them as environment variables.

    Args:
        mlflow_credentials_file (str): The path to the MLflow credentials file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    with open(os.path.expanduser(mlflow_credentials_file)) as f:
        for line in f:
            key, value = line.strip().split('=')
            key = key.split(' ')[-1]
            value = value.replace("'", "").replace('"', '')
            os.environ[key] = value


def get_config_from_mlflow(experiment_name, tracking_uri, run_idx=0):
    """
    Retrieves the configuration file from MLflow for a given experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        run_idx (int, optional): The index of the run to retrieve the configuration from. Defaults to 0.

    Returns:
        dict: The configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Get the experiment and enumerate the runs that have been executed
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_id)

    # Assumes first run in list is the last one that was executed
    run = runs.loc[run_idx]

    # Find all artifacts
    artifact_repo = mlflow.artifacts.get_artifact_repository(run.artifact_uri)
    all_artifacts = artifact_repo.list_artifacts()

    # find the artifacts that have the name `config_*.json`
    config_artifacts = [a for a in all_artifacts if 'config_' in a.path]
    if not config_artifacts:
        raise FileNotFoundError("Configuration file not found.")

    config = config_artifacts[-1]

    # download the config file
    temp_dir = tempfile.mkdtemp()
    artifact_repo.download_artifacts(config.path, temp_dir)
    config_path = os.path.join(temp_dir, config.path)
    with open(config_path) as f:
        config = json.load(f)
    return config


def try_get_checkpoint(
    experiment_name,
    tracking_uri,
    checkpoint_dir='.',
    run_idx=0
):
    """
    Try to get the latest checkpoint from the database, if that fails, get it from the local logs.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        run_idx (int, optional): The index of the run to retrieve the checkpoint from. Defaults to 0.

    Returns:
        str: The path to the latest checkpoint file.
    """
    try:
        model_weights_file = get_checkpoint_from_database(
            experiment_name,
            tracking_uri,
            run_idx,
        )
    except:
        model_weights_file = get_checkpoint_from_local_logs(
            experiment_name,
            tracking_uri,
            checkpoint_dir,
        )
    return model_weights_file


def get_checkpoint_from_database(
    experiment_name,
    tracking_uri,
    run_idx=0,
):
    """
    Retrieves the latest checkpoint from the specified MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        run_idx (int, optional): The index of the run to retrieve the checkpoint from. Defaults to 0.

    Returns:
        str: The path to the latest checkpoint file.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_id)
    # Assumes first run in list is the last one that was executed
    run = runs.loc[run_idx]
    artifact_repo = mlflow.artifacts.get_artifact_repository(run.artifact_uri)
    all_artifacts = artifact_repo.list_artifacts()
    model_artifacts = [a for a in all_artifacts if a.path.startswith("model")]
    checkpoint_artifact = model_artifacts[-1]
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, "checkpoint")
    os.makedirs(checkpoint_dir)

    # download the checkpoint
    artifact_repo.download_artifacts(checkpoint_artifact.path, checkpoint_dir)

    # list the downloaded files including walking the directories
    downloaded_checkpoints = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".ckpt"):
                downloaded_checkpoints.append(
                    os.path.join(root, file)
                )

    latest_checkpoint = downloaded_checkpoints[-1]
    return latest_checkpoint


def get_checkpoint_from_local_logs(
    experiment_name,
    tracking_uri,
    log_dir,
):
    """
    Get the latest checkpoint file from the local logs directory.

    Args:
        experiment_name (str): The name of the experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        log_dir (str): The directory where the logs are stored.

    Returns:
        str: The path to the latest checkpoint file.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    log_dir = f'{os.path.abspath(log_dir)}/{experiment_id}'
    checkpoints = glob(f'{log_dir}/**/*.ckpt', recursive=True)
    checkpoints = sorted(checkpoints, key=os.path.getmtime)
    resume_checkpoint = checkpoints[-1]
    return resume_checkpoint

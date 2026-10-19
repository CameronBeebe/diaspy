"""MLflow dashboard setup and utilities.

DSPy has native MLflow integration via dspy.GEPA(use_mlflow=True).
This script sets up MLflow tracking and provides utilities.
"""
import os
from pathlib import Path


def setup_mlflow(experiment_name: str = "theory-driven-gepa", tracking_uri: str = None):
    """Set up MLflow tracking.
    
    Args:
        experiment_name: Name for MLflow experiment
        tracking_uri: MLflow tracking URI (defaults to ./mlruns)
    """
    import mlflow
    
    if tracking_uri is None:
        # Use local directory
        tracking_uri = f"file://{Path.cwd() / 'mlruns'}"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    
    return tracking_uri


def launch_mlflow_ui(tracking_uri: str = None, host: str = "127.0.0.1", port: int = 5000):
    """Launch MLflow UI server.
    
    Run this in a separate terminal or tmux session:
        python3 -c "from experiments.mlflow_utils import launch_mlflow_ui; launch_mlflow_ui()"
    
    Or directly:
        mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
    """
    import subprocess
    
    if tracking_uri is None:
        tracking_uri = str(Path.cwd() / "mlruns")
    else:
        tracking_uri = tracking_uri.replace("file://", "")
    
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", tracking_uri,
        "--host", host,
        "--port", str(port)
    ]
    
    print(f"Starting MLflow UI at http://{host}:{port}")
    print(f"Backend: {tracking_uri}")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped MLflow UI")


def log_run_to_mlflow(result, parent_run_id=None):
    """Log a single run result to MLflow.
    
    Args:
        result: RunResult instance
        parent_run_id: Parent MLflow run ID for nested runs
    """
    import mlflow
    
    run_name = f"{result.arm}_N{result.n_labels}_seed{result.seed}"
    
    with mlflow.start_run(run_name=run_name, nested=bool(parent_run_id)):
        # Log parameters
        mlflow.log_param("arm", result.arm)
        mlflow.log_param("n_labels", result.n_labels)
        mlflow.log_param("seed", result.seed)
        mlflow.log_param("train_size", result.train_size)
        mlflow.log_param("val_size", result.val_size)
        
        # Log metrics
        if result.test_accuracy is not None:
            mlflow.log_metric("test_accuracy", result.test_accuracy)
        if result.val_accuracy is not None:
            mlflow.log_metric("val_accuracy", result.val_accuracy)
        
        # Budget metrics
        mlflow.log_metric("program_rollouts", result.program_rollouts)
        mlflow.log_metric("reflection_calls", result.reflection_calls)
        mlflow.log_metric("reflection_tokens", result.reflection_tokens)
        
        # Success/failure
        mlflow.log_metric("optimizer_success", float(result.optimizer_success))
        mlflow.log_metric("within_budget", float(result.within_budget_tolerance))
        
        if result.runtime_seconds:
            mlflow.log_metric("runtime_seconds", result.runtime_seconds)
        
        # Tags
        mlflow.set_tag("arm", result.arm)
        mlflow.set_tag("status", "success" if result.optimizer_success else "failed")


def export_results_to_mlflow(results_dir: Path, experiment_name: str = "theory-driven-gepa"):
    """Load results from store and log to MLflow.
    
    Useful for post-hoc analysis if runs weren't logged during execution.
    """
    from experiments.store import ResultsStore
    
    setup_mlflow(experiment_name)
    
    store = ResultsStore(results_dir)
    results = list(store.load_runs())
    
    print(f"Exporting {len(results)} runs to MLflow...")
    
    for i, result in enumerate(results):
        log_run_to_mlflow(result)
        if (i + 1) % 10 == 0:
            print(f"  Logged {i + 1}/{len(results)} runs")
    
    print(f"✓ Exported {len(results)} runs to MLflow")
    print(f"\nView results: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        # Launch UI
        launch_mlflow_ui()
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        # Export results from store
        results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
        export_results_to_mlflow(results_dir)
    else:
        print("Usage:")
        print("  python3 experiments/mlflow_utils.py ui              # Launch MLflow UI")
        print("  python3 experiments/mlflow_utils.py export [dir]    # Export results to MLflow")

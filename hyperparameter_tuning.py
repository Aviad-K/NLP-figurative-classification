import optuna
import subprocess
import json
import os
import uuid
import sys
import shutil

# Define a constant for the consolidated results file
TUNING_RESULTS_LOG_FILE = "tuning_results.jsonl"

def objective(trial):
    """
    Define the objective function for Optuna to optimize.
    Each trial will train a model with a different set of hyperparameters.
    """
    # --- Define Hyperparameter Search Space ---
    
    # 1. Categorical parameters
    per_device_train_batch_size = 32
    n_splits = 5
    
    # 2. Integer parameters - Range adjusted based on previous run.
    pos_embedding_dim = trial.suggest_int("pos_embedding_dim", 2, 16)
    fgpos_embedding_dim = trial.suggest_int("fgpos_embedding_dim", 0, 8)
    
    # 3. Float parameters (log scale) - Range adjusted based on previous run.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    
    # 4. Float parameters (uniform scale)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    # Range adjusted based on previous run.
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.4)

    # --- Construct the Training Command ---
    
    # Use a unique but predictable temporary filename for this trial's result.
    # This file will be deleted after its contents are logged.
    temp_results_filename = f"temp_trial_results_{trial.number}.json"
    
    # Get the path to the python executable from the current environment
    python_executable = sys.executable

    # The base model to fine-tune.
    model_name = "roberta-large"
    print(f"Starting from base model: '{model_name}'.")

    command = [
        python_executable, "train.py",
        "--model_name", model_name,
        "--do_train",
        "--compute_metrics",
        "--n_splits", str(n_splits),
        "--num_epochs", "3",
        "--dataloader_num_workers", "0",
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--per_device_eval_batch_size", str(per_device_train_batch_size),
        "--pos_embedding_dim", str(pos_embedding_dim),
        "--fgpos_embedding_dim", str(fgpos_embedding_dim),
        "--learning_rate", f"{learning_rate:.8f}",
        "--weight_decay", f"{weight_decay:.8f}",
        "--warmup_ratio", f"{warmup_ratio:.8f}",
        "--results_file", temp_results_filename,
        "--output_dir", f"./tuning_output/trial_{trial.number}"
    ]

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: {trial.params}")
    print(f"  Command: {' '.join(command)}")

    # --- Run the Training Script ---
    
    try:
        # Run the training script as a subprocess, streaming output directly
        subprocess.run(command, check=True, text=True)
        
        # --- Read the Objective Value ---
        with open(temp_results_filename, 'r') as f:
            results = json.load(f)
        
        f1_score = results.get("mean_eval_f1", -1.0)

        # --- Log results to the consolidated file ---
        log_entry = {
            "trial_number": trial.number,
            "f1_score": f1_score,
            "params": trial.params,
        }
        with open(TUNING_RESULTS_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Optuna's goal is to maximize this value
        return f1_score

    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error. See console output above for details.")
        # If the trial fails, log it and return a very low value
        log_entry = {
            "trial_number": trial.number,
            "f1_score": -1.0,
            "params": trial.params,
            "error": "CalledProcessError: See console output for details.",
        }
        with open(TUNING_RESULTS_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        return -1.0
    except FileNotFoundError:
        print(f"Trial {trial.number} failed: Could not find results file {temp_results_filename}.")
        # Log the failure
        log_entry = {
            "trial_number": trial.number,
            "f1_score": -1.0,
            "params": trial.params,
            "error": "FileNotFoundError",
        }
        with open(TUNING_RESULTS_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        return -1.0
    finally:
        # Clean up the temporary results file
        if os.path.exists(temp_results_filename):
            os.remove(temp_results_filename)

def save_and_cleanup_callback(study, trial):
    """
    A callback to save the best model and clean up trial artifacts on the fly
    to conserve disk space.
    """
    best_model_dir = "./tuning_output/best_model"
    trial_model_dir = f"./tuning_output/trial_{trial.number}"

    # If the current trial is the best, save its model
    if trial.value is not None and trial.value >= study.best_value:
        print(f"Callback: Trial {trial.number} is the new best with F1: {trial.value:.4f}. Saving model.")
        
        # Remove old best model
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
            
        # Copy new best model
        try:
            if os.path.exists(trial_model_dir):
                shutil.copytree(trial_model_dir, best_model_dir)
                print(f"Callback: Saved new best model to '{best_model_dir}'.")
        except OSError as e:
            print(f"Callback: Error copying model for trial {trial.number}: {e}")

    # Clean up the current trial's model directory immediately after it's done.
    if os.path.exists(trial_model_dir):
        try:
            shutil.rmtree(trial_model_dir)
            print(f"Callback: Cleaned up directory '{trial_model_dir}'.")
        except OSError as e:
            print(f"Callback: Error cleaning up directory for trial {trial.number}: {e}")

if __name__ == "__main__":
    best_params_from_last_run = None
    # --- Read previous results to find the best trial for a warm start ---
    if os.path.exists(TUNING_RESULTS_LOG_FILE):
        print(f"Reading previous results from {TUNING_RESULTS_LOG_FILE} for warm start...")
        best_f1 = -1.0
        with open(TUNING_RESULTS_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('f1_score', -1.0) > best_f1:
                        best_f1 = data['f1_score']
                        best_params_from_last_run = data.get('params')
                except json.JSONDecodeError:
                    continue # Ignore malformed lines
        
        if best_params_from_last_run:
            print(f"Found best parameters from last run (F1: {best_f1:.4f}). Enqueuing for the first trial.")
            print(f"  Params: {best_params_from_last_run}")
    # --- Clean up previous run's log file and temporary trial folders ---
    if os.path.exists(TUNING_RESULTS_LOG_FILE):
        os.remove(TUNING_RESULTS_LOG_FILE)
        print(f"Removed old log file: {TUNING_RESULTS_LOG_FILE}")
    
    # Clean up old trial folders, but preserve the best_model directory
    tuning_dir = "./tuning_output"
    if os.path.exists(tuning_dir):
        for item in os.listdir(tuning_dir):
            item_path = os.path.join(tuning_dir, item)
            if os.path.isdir(item_path) and item != "best_model":
                print(f"Removing old trial directory: {item_path}")
                shutil.rmtree(item_path)
    else:
        os.makedirs(tuning_dir)


    # --- Create and Run the Optuna Study ---
    
    # We want to maximize the F1 score, so the direction is 'maximize'
    study = optuna.create_study(direction="maximize")

    # Enqueue the best trial from the last run if it was found
    if best_params_from_last_run:
        study.enqueue_trial(best_params_from_last_run)
    
    # Start the optimization process with the callback
    study.optimize(objective, n_trials=50, callbacks=[save_and_cleanup_callback])

    # --- Final Cleanup: Remove all individual trial model folders ---
    print("\n--- Cleaning up temporary trial model directories ---")
    try:
        for item in os.listdir("./tuning_output"):
            item_path = os.path.join("./tuning_output", item)
            if os.path.isdir(item_path) and item != "best_model":
                shutil.rmtree(item_path)
        print("Cleanup complete. Only the 'best_model' directory remains.")
    except FileNotFoundError:
        print("Tuning output directory not found, skipping final cleanup.")


    # --- Print the Results ---
    
    print("\n\n--- Hyperparameter Tuning Finished ---")
    print(f"Results have been saved to {TUNING_RESULTS_LOG_FILE}")
    if os.path.exists("./tuning_output/best_model"):
        print(f"The best model has been saved to ./tuning_output/best_model")
    print(f"Number of finished trials: {len(study.trials)}")

    if study.best_trial:
        print("\nBest trial:")
        best_trial = study.best_trial
        print(f"  Value (F1 Score): {best_trial.value:.4f}")

        print("\n  Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("\nNo successful trials were completed.")


    # # You can also visualize the results if you have plotly installed:
    # # `pip install plotly`
    # try:
    #     import plotly
    #     if study.trials:
    #         fig = optuna.visualization.plot_optimization_history(study)
    #         fig.show()
            
    #         fig = optuna.visualization.plot_param_importances(study)
    #         fig.show()
    # except ImportError:
    #     print("\nInstall plotly to visualize optimization history and parameter importances: `pip install plotly`")
    # except Exception as e:
    #     print(f"Could not generate plots: {e}")


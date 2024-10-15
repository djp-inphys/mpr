import warnings
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss
from config_submit import config, clsfrs, clsfrs_params
from src.beale import BealeAlgorithm

warnings.filterwarnings("ignore", category=FutureWarning)


import logging
from typing import Optional, List
import matplotlib.pyplot as plt


def call_beale(descs: np.ndarray, clsvec: np.ndarray, n_selvar: Optional[int] = None) -> List[int]:
    """
    Call Beale's regression reselection algorithm to select variables.

    Args:
        descs (np.ndarray): Descriptor matrix.
        clsvec (np.ndarray): Class vector.
        n_selvar (Optional[int]): Number of variables to select. If None, uses config value.

    Returns:
        List[int]: Indices of selected variables.
    """
    try:
        clsarr = clsvec.reshape(-1, 1)
        combined_space = np.hstack((descs, clsarr))
        correlation_matrix = np.corrcoef(combined_space, rowvar=False)

        if n_selvar is None:
            n_selvar = config['n_selvar']

        selected_variables = beale(correlation_matrix, n_selvar)
        variable_indices = np.arange(descs.shape[1])
        return variable_indices[selected_variables == 1].tolist()
    except Exception as e:
        logging.error(f"Error in call_beale: {str(e)}")
        raise

def train_models(models, dataset, labels):
    """
    Train each model in the list on the provided dataset.
    """
    try:
        trained_models = []
        for i, model in enumerate(models):
            logging.debug(f"Training model {i+1} of {len(models)}")
            trained_model = model.fit(dataset, labels)
            trained_models.append(trained_model)
            logging.debug(f"Model {i+1} training completed")
        
        logging.info(f"All {len(models)} models have been trained")
        return trained_models
    except Exception as e:
        print(e)


def classify(models, instance):
    """
    Predict probabilities for an instance using a list of classifiers.
    """
    try:
        probabilities = []
        instance_array = np.array(instance).reshape(1, -1)  # Reshape to 2D array
        for i, model in enumerate(models):
            prob = model.predict_proba(instance_array)[0]  # Get probability for class 0
            probabilities.append(prob)
            logging.debug(f"Model {i}: Probability for class 0 = {prob}")
        return probabilities
    except Exception as e:
        print(e)


def classifiers(models, unk_vec):
    try:
        resvec = [];
        for clsno in range(len(models)):
            probs = models[clsno].predict_proba(unk_vec)
            resvec.append(probs[:, 0])

        return resvec
    except Exception as e:
        print(e)

def leave_one_out(unk_idx, models, dataset, labels, results):
    """
    Perform leave-one-out cross-validation for a specific instance.
    """
    try:
        data_without_unknown = np.delete(dataset, unk_idx, axis=0)
        labels_without_unknown = np.delete(labels, unk_idx)
        
        # Train models without the unknown instance
        trained_models = train_models(models, data_without_unknown, labels_without_unknown)
        
        # Classify the unknown instance
        unknown_instance = np.array(dataset[unk_idx, :])
        results[unk_idx, :] = classify(trained_models, unknown_instance)
    except Exception as e:
        print(e)


def process_parallel(function, iterable):
    """
    Perform parallel processing using a thread pool.
    """
    pool = ThreadPool(config["worker_pool_size"])
    results = pool.map(function, iterable)
    pool.close()
    pool.join()

    return results


def process_none_parallel(dataset, labels):
    """
    Main process to train classifiers and apply them on a leave-one-out basis.
    """
    try:
        num_unknowns, num_features = dataset.shape
        results = np.zeros((num_unknowns, len(clsfrs)))
        models = []
        
        # Stratified K-fold cross-validation setup
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train models using hyperparameter optimization
        for key, classifier in clsfrs.items():
            logging.info(f"Training classifier: {key}")
            model = classifier  # Use the classifier directly
            
            param_grid = clsfrs_params.get(key, {})
            
            if param_grid:  # If there are parameters to tune
                logging.info(f"Performing grid search for {key} with parameters: {param_grid}")
                grid_search = GridSearchCV(model, param_grid=param_grid, cv=folds, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(dataset, labels)
                model = grid_search.best_estimator_
                logging.info(f"Best parameters for {key}: {grid_search.best_params_}")
            else:
                logging.info(f"No parameter grid for {key}. Fitting model directly.")
                model.fit(dataset, labels)
            
            models.append(model)
            logging.info(f"Finished training {key}")
        
        
        # Perform leave-one-out cross-validation
        logging.info("Dataset visualization saved as 'dataset_visualization.png'")
        for unk_idx in range(num_unknowns):
            leave_one_out(unk_idx, models, dataset, labels, results)
        
        # Variable selection using Beale's method
        best_log_loss = float('inf')
        best_selected_indices = []

        for n_selvar in range(3, len(clsfrs)):
            selected_indices = call_beale(results, labels, n_selvar)
            linear_reg = lm.LinearRegression()
            linear_reg.fit(results[:, selected_indices], labels.reshape(-1, 1))
            predictions = linear_reg.predict(results[:, selected_indices])

            current_log_loss = log_loss(labels, predictions)
            
            if current_log_loss < best_log_loss:
                best_selected_indices = selected_indices
                best_log_loss = current_log_loss

        # Use the best selected models
        final_models = [models[i] for i in best_selected_indices]
        
        logging.info(f"Process completed successfully. Best log loss: {best_log_loss}")
        logging.debug(f"Best selected indices: {best_selected_indices}")
        
    except Exception as e:
        logging.error(f"Error in process_none_parallel: {str(e)}")
        raise

    return results[:, best_selected_indices], final_models, best_selected_indices

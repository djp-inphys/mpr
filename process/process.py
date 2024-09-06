import warnings
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss
from beale import beale
from config_submit import config, clsfrs, clsfrs_params

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def call_beale(descs, clsvec, n_selvar=None):
    """
    Call Beale's regression reselection algorithm to select variables.
    """
    clsarr = clsvec.reshape(-1, 1)
    combined_space = np.hstack((descs, clsarr))
    correlation_matrix = np.corrcoef(combined_space, rowvar=False)

    if n_selvar is None:
        n_selvar = config['n_selvar']

    selected_variables = beale(correlation_matrix, n_selvar)
    variable_indices = np.arange(descs.shape[1])
    return variable_indices[selected_variables == 1]


def train_models(models, dataset, labels):
    """
    Train each model in the list on the provided dataset.
    """
    try:
        trained_models = [model.fit(dataset, labels) for model in models]
        return trained_models
    except Exception as e:
        print(e)


def classify(models, instance):
    """
    Predict probabilities for an instance using a list of classifiers.
    """
    try:
        return [model.predict_proba(instance)[:, 0] for model in models]
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
        unknown_instance = np.matrix(dataset[unk_idx, :])
        results[unk_idx, :] = classify(trained_models, unknown_instance)
    except Exception as e:
        print(e)


def parallel_processing(function, iterable):
    """
    Perform parallel processing using a thread pool.
    """
    pool = ThreadPool(config["worker_pool_size"])
    results = pool.map(function, iterable)
    pool.close()
    pool.join()
    return results


def process(dataset, labels):
    """
    Main process to train classifiers and apply them on a leave-one-out basis.
    """
    try:
        models = []
        num_unknowns, num_features = dataset.shape
        results = np.zeros((num_unknowns, len(clsfrs)))
        n_folds = 20

        # Stratified K-fold cross-validation setup
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Train models using hyperparameter optimization
        for key, classifier_str in clsfrs.items():
            model = eval(classifier_str)
            param_grid = clsfrs_params[key]
            
            if param_grid:
                grid_search = GridSearchCV(model, param_grid=param_grid, cv=folds, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(dataset, labels)
                print(f'Best score: {grid_search.best_score_}')
                print(f'Best parameters: {grid_search.best_params_}')
                model.set_params(**grid_search.best_params_)
            
            models.append(model)
        
        # Perform leave-one-out cross-validation
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
            print(current_log_loss)

            if current_log_loss < best_log_loss:
                best_selected_indices = selected_indices
                best_log_loss = current_log_loss

        # Train models using the best selected variables
        models_array = np.array(models)
        final_models = train_models(models_array[best_selected_indices], dataset, labels)

    except Exception as e:
        print(e)

    return results[:, best_selected_indices], final_models, best_selected_indices

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer

from configs.model_configs import model_configs
from configs.preprocessing_configs import preprocessing_configs

# sklearn uses np.random. Keep in mind that np.random is not threadsafe
np.random.seed(3141592)

if __name__ == "__main__":
    # # Parse arguments, setup the experiment folder, load the configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc', '--preprocessing_config', required=True, type=str,
                        help="Name of the preprocessing config in configs/preprocessing_configs "\
                            "used for this experiment.")
    parser.add_argument('-mc', '--model_config', required=True, type=str,
                        help="Name of the model config in configs/model_configs "\
                            "used for this experiment.")
    parser.add_argument('-cv', '--cv_folds', type=int, default=5, help="Number of folds to use in CV.")
    parser.add_argument('-nj', '--n_jobs', type=int, default=1, 
                        help="Number of jobs used for grid search. More than 1 uses multithreading.")
    args = parser.parse_args()

    if args.preprocessing_config not in preprocessing_configs:
        raise ValueError(f"Unkown preprocessing config {args.preprocessing_config}.")
    if args.model_config not in model_configs:
        raise ValueError(f"Unkown model config {args.model_config}.")

    experiment_name = f"{args.preprocessing_config}---{args.model_config}"

    if os.path.isdir(f"results/{experiment_name}"):
        raise Exception(f"Results for experiment {experiment_name} already exist. Delete the " \
                        f"results/{experiment_name} folder to be able to redo the experiment.")

    p_config = preprocessing_configs[args.preprocessing_config]
    m_config = model_configs[args.model_config]


    # # Load the data
    X_train = pd.read_csv('data/X_train.csv', index_col='id')
    y_train = pd.read_csv('data/y_train.csv', index_col='id')


    # # Create the Preprocessing + Model pipeline
    # Add preprocessing steps to the pipeline
    pipeline_steps = [(step, p_config[f'{step}'](**p_config[f'{step}_hyperparams'])) for step in p_config['order']]

    # Add model to the pipeline
    model = m_config['model'](**m_config['model_hyperparams'])
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(steps=pipeline_steps)


    # # Grid Search CV
    param_grid = p_config['param_grid']
    param_grid.update(m_config['param_grid'])  # Combine param_grids from p_config and m_config
    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=args.n_jobs, 
                                scoring=make_scorer(r2_score), return_train_score=True,
                                verbose=3, cv=args.cv_folds)
    grid_search.fit(X_train, np.ravel(y_train))

    # # Process the results results
    print(f"Best parameters (CV score={grid_search.best_score_:0.4f}):")
    print(grid_search.best_params_)
    cv_results = grid_search.cv_results_
    cv_results['best_score'] = grid_search.best_score_
    cv_results['best_params'] = grid_search.best_params_
    print(cv_results)

    # Make results json serializable
    for key in cv_results:
        if isinstance(cv_results[key], np.ndarray):
            cv_results[key] = cv_results[key].tolist()


    # # Save the results
    print(f"Saving results to results/{experiment_name}")
    os.mkdir(f'results/{experiment_name}')
    config = {
        'preprocessing_config': p_config,
        'model_config': m_config,
        'n_jobs': args.n_jobs,
        'cv_folds': args.cv_folds
    }
    with open(f'results/{experiment_name}/config.json', 'w') as f:
        json.dump(config, f, default=lambda f: f.__name__)
    with open(f'results/{experiment_name}/results.json', 'w') as f:
        json.dump(grid_search.cv_results_, f, default=lambda f: f.__name__)


    # # Generate test set prediction
    X_test = pd.read_csv('data/X_test.csv', index_col='id')
    y_test = grid_search.predict(X_test)
    y_test = pd.DataFrame(y_test, columns=['y'])
    y_test.to_csv(f'results/{experiment_name}/y_test.csv', index_label='id')

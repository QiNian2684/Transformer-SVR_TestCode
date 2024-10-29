import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import optuna


def main():
    train_times = 200

    print("=== Indoor Location Prediction with SVR and Optuna ===\n")

    # Load data
    print("Loading training and validation data...")
    train_data = pd.read_csv('UJIndoorLoc/trainingData.csv')  # Updated file path to include all buildings
    validation_data = pd.read_csv('UJIndoorLoc/validationData.csv')  # Updated file path to include all buildings
    print("Data loaded successfully.\n")

    # Optional: Display basic information about the datasets
    print("Training Data Shape:", train_data.shape)
    print("Validation Data Shape:", validation_data.shape, "\n")

    # Separate features and targets
    print("Separating features and targets...")
    # Adjust the columns to drop based on your dataset's actual structure
    features_train = train_data.drop(columns=[
        'LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID',
        'SPACEID', 'RELATIVEPOSITION', 'USERID',
        'PHONEID', 'TIMESTAMP'
    ])
    target_train = train_data.loc[:, ['LONGITUDE', 'LATITUDE']]

    features_val = validation_data.drop(columns=[
        'LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID',
        'SPACEID', 'RELATIVEPOSITION', 'USERID',
        'PHONEID', 'TIMESTAMP'
    ])
    target_val = validation_data.loc[:, ['LONGITUDE', 'LATITUDE']]
    print("Features and targets separated.\n")

    # Define objective function for Optuna
    def objective(trial):
        print(f"--- Starting Trial {trial.number + 1} ---")

        # Suggest hyperparameters using suggest_float with log=True
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-3, 1e1, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
        else:
            gamma = 'scale'

        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
        else:
            degree = 3  # default degree

        print(
            f"Trial {trial.number + 1} parameters: C={C:.4f}, epsilon={epsilon:.4f}, kernel={kernel}, gamma={gamma}, degree={degree}")

        # Train SVR models for LONGITUDE and LATITUDE
        svr_lon = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)
        svr_lat = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)

        svr_lon.fit(features_train, target_train['LONGITUDE'])
        svr_lat.fit(features_train, target_train['LATITUDE'])

        print("Models trained.")

        # Predict on validation data
        print("Making predictions on validation data...")
        pred_lon = svr_lon.predict(features_val)
        pred_lat = svr_lat.predict(features_val)
        print("Predictions completed.")

        # Calculate errors
        errors = np.sqrt((pred_lon - target_val['LONGITUDE']) ** 2 + (pred_lat - target_val['LATITUDE']) ** 2)
        mean_error = np.mean(errors)
        median_error = np.median(errors)

        print(
            f"Trial {trial.number + 1} results: Mean Error = {mean_error:.2f} meters, Median Error = {median_error:.2f} meters\n")

        # Return mean error as the objective to minimize
        return mean_error

    # Optimize hyperparameters
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=train_times, show_progress_bar=True)
    print("Hyperparameter optimization completed.\n")

    # Get best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print()

    # Extract best hyperparameters
    best_C = best_params['C']
    best_epsilon = best_params['epsilon']
    best_kernel = best_params['kernel']
    best_gamma = best_params.get('gamma', 'scale')
    best_degree = best_params.get('degree', 3)

    # Train models with best hyperparameters
    print("Training final models with best hyperparameters...")
    svr_lon = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel, gamma=best_gamma, degree=best_degree)
    svr_lat = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel, gamma=best_gamma, degree=best_degree)

    svr_lon.fit(features_train, target_train['LONGITUDE'])
    svr_lat.fit(features_train, target_train['LATITUDE'])
    print("Final models trained.\n")

    # Predict on validation data
    print("Making final predictions on validation data...")
    pred_lon = svr_lon.predict(features_val)
    pred_lat = svr_lat.predict(features_val)
    print("Final predictions completed.\n")

    # Calculate errors
    print("Calculating error metrics...")
    errors = np.sqrt((pred_lon - target_val['LONGITUDE']) ** 2 + (pred_lat - target_val['LATITUDE']) ** 2)
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    # Calculate MSE
    mse_lon = mean_squared_error(target_val['LONGITUDE'], pred_lon)
    mse_lat = mean_squared_error(target_val['LATITUDE'], pred_lat)

    # Predicted and actual coordinates
    predictions = np.vstack((pred_lon, pred_lat)).T
    actuals = target_val[['LONGITUDE', 'LATITUDE']].values

    # Calculate Euclidean distances
    differences = np.sqrt(np.sum((predictions - actuals) ** 2, axis=1))

    # Calculate average 2D error
    average_2d_error = np.mean(differences)

    print("Error metrics calculated.\n")

    # Print evaluation results
    print("=== Evaluation Metrics ===")
    print(f"MSE for Longitude: {mse_lon:.2f}")
    print(f"MSE for Latitude: {mse_lat:.2f}")
    print(f"Mean Error: {mean_error:.2f} meters")
    print(f"Median Error: {median_error:.2f} meters")
    print(f"Average 2D Error: {average_2d_error:.2f} meters\n")

    # Optionally, display some sample predictions
    print("=== Sample Predictions ===")
    sample_size = 5
    for i in range(sample_size):
        print(f"Sample {i + 1}:")
        print(f"  Predicted Longitude: {pred_lon[i]:.2f}, Actual Longitude: {target_val['LONGITUDE'].iloc[i]:.2f}")
        print(f"  Predicted Latitude: {pred_lat[i]:.2f}, Actual Latitude: {target_val['LATITUDE'].iloc[i]:.2f}")
        print(f"  Error: {errors[i]:.2f} meters\n")


if __name__ == "__main__":
    main()

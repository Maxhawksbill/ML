from typing import List, Tuple

import numpy as np
import pandas as pd


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point.
        point2 (np.ndarray): Second point.

    Returns:
        float: Euclidean distance between two points.
    """
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


class KNN:
    """K Nearest Neighbors classifier."""

    def __init__(self, k: int) -> None:
        """Initialize KNN with the number of neighbors to consider (k).

        Args:
            k (int): Number of neighbors to consider.
        """
        self._X_train = None
        self._y_train = None
        self.k = k  # number of neighbors to consider

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the KNN model with training data.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target.
        """
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Predict target values for test data.

        Args:
            X_test (np.ndarray): Test data features.
            verbose (bool, optional): Print progress during prediction. Defaults to False.

        Returns:
            np.ndarray: Predicted target values.
        """
        n = X_test.shape[0]
        y_pred = np.empty(n, dtype=self._y_train.dtype)

        for i in range(n):
            distances = np.array([euclidean_distance(x, X_test[i]) for x in self._X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self._y_train[k_indices]
            y_pred[i] = np.bincount(k_nearest_labels).argmax()

            if verbose:
                print(f"Predicted {i+1}/{n} samples", end="\r")

        if verbose:
            print("")
        return y_pred


def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split dataset into k folds for cross-validation.

    Args:
        X (np.ndarray): Dataset features.
        y (np.ndarray): Dataset target.
        k (int): Number of folds.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: List of tuples (X_train, y_train, X_test, y_test).
    """
    n_samples = X.shape[0] # we take only the number of rows
    fold_size = n_samples // k
    folds = []  # Container to store the results of each fold

    # Split the data into k folds
    for i in range(k):
        X_train = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]]) # concatenate the arrays for training data features
        y_train = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]]) # concatenate the arrays for training target data
        X_val = X[i * fold_size:(i + 1) * fold_size]
        y_val = y[i * fold_size:(i + 1) * fold_size]
        folds.append((X_train, y_train, X_val, y_val))

    return folds
    


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Accuracy score.
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy


def main() -> None:
    """Main function to demonstrate the KNN classifier and k-fold cross-validation."""
    # Read training and testing data from CSV files
    path_train = "HW2_KNN/data/train.csv"
    path_test = "HW2_KNN/data/test.csv"
    training_data = pd.read_csv(path_train)[:1000] # take only 1000 samples for faster computation
    testing_data = pd.read_csv(path_test) # take all samples for testing

    # Extract features and target from the training data
    X = training_data.iloc[:, 1:].values
    y = training_data.iloc[:, 0].values 
    print("Training data:", X.shape, y.shape)

    # Extract features and target from the testing data
    X_test = testing_data.iloc[:, 1:].values
    y_test = testing_data.iloc[:, 0].values
    print("Test data:", X_test.shape, y_test.shape)

    k = 10 # number of neighbors to consider
    print(f" KNN with k = {k}")

    num_folds = 8

    # Perform k-fold cross-validation
    for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val, verbose=True)
        accuracy = evaluate_accuracy(y_val, y_pred)
        print(f"Accuracy: {round(accuracy, 2)}")

    # For the different values of k in KNN, calculate the accuracy on the test dataset

    # Create a dataframe to store the results
    results = pd.DataFrame(columns=["k", "Accuracy"])
    results["k"] = [3, 4, 5, 6, 7, 9, 10, 15, 20, 21, 40, 41]

    for k in results["k"]:
        accuracies = []
        for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
            model = KNN(k=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val, verbose=False)
            accuracies.append(evaluate_accuracy(y_val, y_pred))
        results.loc[results["k"] == k, "Accuracy"] = round(np.mean(accuracies), 2)

    print(results)

    # Evaluate the best model on the test data
    best_k = results.loc[results["Accuracy"].idxmax(), "k"]
    print(f"Best k from cross-validation: {best_k}")

    model = KNN(k=best_k)
    model.fit(X, y)
    y_test_pred = model.predict(X_test)
    test_accuracy = evaluate_accuracy(y_test, y_test_pred)
    print(f"Test Accuracy with k={best_k}: {round(test_accuracy, 2)}")


if __name__ == "__main__":
    main()

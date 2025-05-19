# Multi-Class SVM Optimization on UCI Dataset

## 1. Problem Description

This project implements the solution to an assignment focused on optimizing Support Vector Machine (SVM) classifiers for a multi-class dataset. The core requirements were:

1.  **Dataset Selection:** Choose a multi-class dataset from the UCI Machine Learning Repository with a size between 5,000 and 30,000 instances.
2.  **Sampling:** Divide the dataset into training (70%) and testing (30%) sets using 10 different random seeds (creating 10 unique samples or splits).
3.  **SVM Optimization:** For each of the 10 samples, optimize an SVM classifier. The optimization process should involve 100 iterations (interpreted here as 100 hyperparameter combinations evaluated using Randomized Search).
4.  **Reporting:** Report the best hyperparameters found (specifically mentioning Kernel, C, and Gamma, which are relevant for `sklearn.svm.SVC`) and the corresponding test accuracy for each of the 10 samples in a comparative table.
5.  **Convergence Plot:** Generate a convergence graph for the sample that achieved the highest test accuracy. This graph should illustrate the optimization process over the 100 iterations.
6.  **Data Analytics:** Perform and showcase basic data analytics on the selected dataset.
7.  **Showcase:** Present the complete results, code, and analysis on GitHub.

*(Note: The original assignment mentioned 'Nu' and 'Epsilon' parameters, which are typically associated with NuSVC/NuSVR and SVR respectively. This implementation uses the standard `sklearn.svm.SVC` classifier and optimizes its key hyperparameters: `C`, `kernel`, and `gamma`.)*

## 2. Dataset Used: Pen-Based Recognition of Handwritten Digits

* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)
* **Characteristics:**
    * Instances: 10,992 (Combined from `pendigits.tra` and `pendigits.tes`)
    * Attributes: 16 numeric features (representing x, y coordinates from pen movements).
    * Classes: 10 (Digits 0 through 9).
    * Task: Multi-class classification.
* **Reason for Choice:** This dataset fits the size criteria (5k-30k rows) and is a standard multi-class problem. It has easily accessible `.data` files directly from the UCI repository.
* **Initial Dataset Issue:** The "Dry Bean Dataset" was initially considered, but issues were encountered with directly loading the `.xlsx` file from its zipped URL within the script, prompting the switch to the Pen Digits dataset for smoother execution.

## 3. Implementation Details

The project is implemented using a Python script (`svm_optimization.py` - *you can rename the file*) leveraging the following libraries:

* `pandas`: For data loading and manipulation.
* `numpy`: For numerical operations, especially for handling results.
* `scikit-learn`: For machine learning tasks:
    * `train_test_split`: Splitting data.
    * `StandardScaler`: Feature scaling.
    * `SVC`: Support Vector Classifier model.
    * `RandomizedSearchCV`: Hyperparameter optimization.
    * `StratifiedKFold`: Cross-validation strategy ensuring class balance.
    * `accuracy_score`: Evaluating model performance.
* `matplotlib`: For plotting the convergence graph.
* `scipy.stats`: For defining hyperparameter distributions (e.g., `loguniform`).
* `time`, `warnings`, `urllib`: Utility libraries.

**Key Steps in the Script:**

1.  **Load Data:** Reads the `pendigits.tra` and `pendigits.tes` files directly from the UCI URLs using `pandas.read_csv` and concatenates them into a single DataFrame. Defines appropriate column names.
2.  **Basic EDA:** Performs exploratory data analysis by displaying the first few rows (`head()`), dataset information (`info()`), descriptive statistics (`describe()`), and the distribution of classes (`value_counts()`).
3.  **Preprocessing:**
    * Separates features (X) and the target variable (y).
    * The target variable `Class` is already numeric (0-9), so no label encoding is needed.
4.  **Train/Test Splitting & Scaling Loop (10 Samples):**
    * Loops 10 times (`N_SAMPLES=10`).
    * In each iteration, `train_test_split` is called with a different `random_state` (0 to 9) and `stratify=y` to ensure each split maintains the original dataset's class proportions.
    * A `StandardScaler` is fitted **only** on the training data (`X_train`) of the current sample and then used to transform both `X_train` and `X_test`. This prevents data leakage from the test set.
5.  **SVM Hyperparameter Optimization:**
    * Inside the loop, an `SVC` model is defined.
    * `RandomizedSearchCV` is configured to find the best hyperparameters:
        * `estimator`: The `SVC` model.
        * `param_distributions`: A dictionary defining the search space for `C`, `gamma`, and `kernel` (using `loguniform` distributions for `C` and `gamma`, focusing on the `rbf` kernel).
        * `n_iter`: Set to `N_ITER_SEARCH = 100`, fulfilling the "100 iterations" requirement. It samples 100 random combinations from the `param_distributions`.
        * `cv`: Uses `StratifiedKFold` (with `CV_FOLDS = 5`) for robust cross-validation within the search, essential for potentially imbalanced multi-class datasets.
        * `scoring`: Uses `'accuracy'` to evaluate combinations.
        * `n_jobs=-1`: Uses all available CPU cores for parallel processing.
        * `random_state`: Ensures reproducibility of the search process for each sample.
    * `random_search.fit(X_train_scaled, y_train)` executes the search.
6.  **Evaluation & Results Storage:**
    * The best estimator found by the search (`random_search.best_estimator_`) is used to make predictions on the scaled test set (`X_test_scaled`).
    * The `accuracy_score` is calculated by comparing predictions (`y_pred`) with the true test labels (`y_test`).
    * The sample number, best accuracy, and the corresponding best parameters (`C`, `kernel`, `gamma`) are stored.
    * The history of mean cross-validated scores during the search (`random_search.cv_results_['mean_test_score']`) is stored for the convergence plot.
7.  **Results Table Generation:** The collected results are compiled into a pandas DataFrame, formatted, and saved as a CSV file (`svm_optimization_results_pendigits.csv`). It is also printed to the console during execution.
8.  **Convergence Plot Generation:**
    * Identifies the sample (`Sample #`) with the highest test accuracy.
    * Retrieves the stored search score history for that best sample.
    * Calculates the running maximum score found up to each iteration using `np.maximum.accumulate`.
    * Uses `matplotlib` to plot both the score of each individual iteration and the best score found so far against the iteration number (1 to 100).
    * The plot is displayed during execution and saved as a PNG file (e.g., `convergence_graph_sample_SX_pendigits.png`).

## 4. Result Table

| Sample     |   Best Accuracy | Best Kernel   |    Best C |   Best Gamma |
|:-----------|----------------:|:--------------|----------:|-------------:|
| S1         |          0.9961 | rbf           |   5.62793 |    0.0473499 |
| S2         |          0.9961 | rbf           |  12.5511  |    0.0594873 |
| S3         |          0.9951 | rbf           | 240.133   |    0.0923696 |
| S4         |          0.9967 | rbf           |   2.35709 |    0.0700782 |
| S5         |          0.9961 | rbf           |   5.55375 |    0.0702962 |
| S6         |          0.9951 | rbf           |  36.2394  |    0.0905468 |
| S7         |          0.9976 | rbf           |  22.2996  |    0.0881138 |
| S8         |          0.9936 | rbf           |   6.95859 |    0.0216093 |
| S9         |          0.9973 | rbf           |  10.1932  |    0.0980293 |
| S10        |          0.9951 | rbf           |  12.0572  |    0.0511758 |


**Discussion:**

* The results table typically shows high accuracy (often >98-99%) for the Pen Digits dataset with an optimized RBF SVM, but highlights potential variability in performance and optimal hyperparameters across different random train-test splits. This underscores the importance of cross-validation and multiple runs for robust model evaluation.
* The convergence plot visualizes how the hyperparameter search progressed. The 'Best Score Found So Far' line usually rises quickly and then plateaus, indicating that the search effectively located a good region in the hyperparameter space within the 100 iterations.

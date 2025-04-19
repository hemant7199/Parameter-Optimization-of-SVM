# Assignment - Parameter Optimization SVM
# Completed By: Hemant
# Roll No. 102217141
# Result Table

![image](https://github.com/user-attachments/assets/584f8cbf-d31c-40eb-9797-6e3da437d20c)

# Result Graph

![image](https://github.com/user-attachments/assets/5b7e8560-b614-4afe-95eb-411627c2fb74)


# Overview

This report offers a comprehensive analysis of the [Dry Bean Dataset](UCI ML Repository ID: 602).

# Methodology

Data Loading: The Dry Bean Dataset is loaded using the UCI ML Repository API.

Data Preprocessing: The dataset is split into features (X) and target (y). A train-test split is performed for model evaluation.

Model Selection: A Nu-Support Vector Classifier (NuSVC) with Bayesian Optimization is employed to optimize hyperparameters.

Model Evaluation: The model's accuracy is evaluated using cross-validation and convergence plots.

Result Analysis: The best hyperparameters and accuracy obtained are analyzed.

# Dataset Description

The Dry Bean Dataset comprises various attributes of dry beans, including geometric, shape, and texture attributes, used for bean classification.

# Fetures
The dataset includes the following features:

Area

Perimeter

MajorAxisLength

MinorAxisLength

AspectRatio

Eccentricity

ConvexArea

EquivDiameter

Extent

Solidity

Roundness

Compactness

ShapeFactor1

ShapeFactor2

ShapeFactor3

ShapeFactor4


# Target

The target variable represents the class of the dry bean.

# Data Exploration

Basic exploratory data analysis is performed:

# Summary Statistics

![image](https://github.com/user-attachments/assets/fe44fe2f-d02e-49c9-bdba-6c60c8face03)







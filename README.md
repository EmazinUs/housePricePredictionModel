
 # House Price Prediction Model

This project is a machine learning-based house price prediction model built using linear regression. The model predicts house prices based on various features such as geographical location, housing age, income levels, and proximity to the ocean.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [How to Run the Project](#how-to-run-the-project)
- [License](#license)

## Project Overview

This project uses a dataset of housing information, which includes various features like geographical coordinates, housing age, total rooms, population, and proximity to the ocean. Using this data, we aim to build a predictive model that estimates the median house value based on these features.

The model was trained using linear regression, and the performance was evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the test set.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## Data Preprocessing

- The dataset was loaded from a CSV file (`housing.csv`).
- Missing values were handled by dropping rows with `NaN` values.
- Outliers were removed by applying the IQR (Interquartile Range) method on `median_house_value` and `median_income`.
- Categorical features (like `ocean_proximity`) were one-hot encoded to create binary features.
- Features such as `total_bedrooms` were dropped due to their high correlation with other features.

## Model Training and Evaluation

1. **Training the Model:**
   - We used linear regression to predict house prices based on selected features.
   - The model was fitted using the training set, and evaluation was performed on the test set.

2. **Performance Metrics:**
   - The model's performance was evaluated using:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)

   The results on the test set:
   - **MSE:** 3,529,059,611.57
   - **RMSE:** 59,405.89

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/EmazinUs/housePricePredictionModel.git

### Instructions for the README:
- The **Technologies Used** section lists all the key libraries used for this project.
- The **Data Preprocessing** section explains how you cleaned and prepared the data before training the model.
- The **Model Training and Evaluation** section provides a brief explanation of how the model was trained and how its performance was evaluated.
- The **How to Run the Project** section gives users the instructions on how to set up and run your project on their own systems.

Feel free to modify the content as necessary based on your project specifics!


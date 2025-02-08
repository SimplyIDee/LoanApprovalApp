# Loan Approval App
This project is a machine learning application that predicts whether a loan will be approved based on user input parameters. The application is built using Streamlit for the web interface and a pre-trained machine learning model for predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Loan Approval Prediction application allows users to input various parameters related to their loan application and receive a prediction on whether their loan is likely to be approved or rejected. The application uses a pre-trained machine learning model to make predictions based on the input data.

## Features
- User-friendly web interface built with Streamlit
- Input fields for various loan application parameters
- Real-time prediction of loan approval status
- Display of user input parameters and prediction result

## Installation
To run this application locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-approval-prediction.git
    cd loan-approval-prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the pre-trained model and scaler files in the project directory:
    - `ApplicantIncome_scaler.pkl`
    - `CoapplicantIncome_scaler.pkl`
    - `LoanAmount_scaler.pkl`
    - `Dependents_encoder.pkl`
    - `PropertyArea_encoder.pkl`
    - `CreditHistory_encoder.pkl`
    - `LoanModel.pkl`

4. Run the Streamlit application:
    ```bash
    streamlit run loanDeploy.py
    ```

## Usage
1. Open the Streamlit application in your web browser.
2. Enter the required loan application parameters in the sidebar.
3. Click the "Push to Predict" button to get the prediction result.
4. The prediction result will be displayed on the main page.

## Model Training
The machine learning model used in this application was trained using a dataset of loan applications. The training process involved the following steps:
1. Data preprocessing and feature engineering
2. Splitting the data into training and testing sets
3. Training a classification model (e.g., Logistic Regression, Random Forest)
4. Evaluating the model's performance on the test set
5. Saving the trained model and scalers for deployment

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

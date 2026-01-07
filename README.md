# Starbucks-Stock-Price-Prediction-Using-Deep-Learning☕
This project explores a deep learning-based system for real-time financial forecasting, focusing on time-series prediction of Starbucks Corporation's daily stock price. Using  data sourced from NASDAQ (spanning from November 13, 2017, to January 10, 2025, comprising 1800 daily samples), the goal was to build and evaluate models that can accurately predict future stock prices.

The core methodology was implementing and comparing four distinct neural network architectures— **Residual MLP, a Vanilla RNN, an LSTM, and a CNN** —to identify the most effective approach for capturing the complex patterns and temporal dependencies in financial data. The project follows a complete machine learning pipeline, from data preprocessing and feature engineering (creating moving averages, lag features, and volatility metrics) to model training, validation, and comprehensive evaluation using metrics like MAE, RMSE, and R².

# Libraries/  Languagaes
-Programming Language: Python

-Core Data & ML Libraries: pandas, numpy, scikit-learn, statsmodels

-Deep Learning Framework: TensorFlow / Keras

-Visualization: matplotlib, seaborn

# The Process
This project implemented a structured deep learning pipeline for time-series forecasting:

*Data Preparation*

Loading the stock data (Open price), parsed dates, and created temporal features (Year, Month, Day).

*Feature Engineering*

Upon reviewsing the EDA, Generated key financial indicators:

-Trend: Simple Moving Averages (3, 7, 30-day).

-Lags: Past price values (3, 7, 30-day lagged features).

-Volatility & Returns: Daily percentage change and rolling standard deviation.

*Sequencing for Time Series*

For RNN/LSTM/CNN models, data was transformed into supervised learning format using a 30-day look-back window (time_steps=30).

*Model Implementation & Training*

Four distinct neural network architectures were built and trained with an 80/10/10 train/validation/test split:

-Residual MLP: A deep feedforward network with residual blocks, L2 regularization, and dropout.

-Vanilla RNN: A basic Recurrent Neural Network to model sequential dependencies.

-LSTM: A Long Short-Term Memory network designed to capture long-range temporal patterns.

-CNN: A 1D Convolutional Neural Network to learn local patterns in the sequence data.

*Evaluation*

Models were evaluated on the test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE/Root of MSE), and R-squared (R²).

# Findings
The Vanilla RNN achieved the best overall performance, with the lowest error (MAE: 0.475) and an exceptional fit to the data (R²: 0.993). For this specific stock price series, a simpler recurrent architecture was sufficient to model the temporal dependencies  effectively and with less overfitting than the more complex LSTM and CNN models.

# WWW
We effectively produced  a rich set of technical indicators (moving averages, lags, returns) provided the models with meaningful patterns to learn from.

Our Time-Series methodology implimented a dedicated train-validation-test split in chronological order and created sequence data (30-day windows), effectively utalising the time-series nature of the data, preventing look-ahead bias.

Building and comparing four different neural network families (MLP, RNN, LSTM, CNN) , presenting valuable insights into which paradigm works best for this specific forecasting problem.

Residual connections, batch normalization, dropout, and L2 regularization in the MLP model demonstrated good practices to combat overfitting in deep feedforward networks.

# EBI

As the models were trained with largely default or manually set parameters, i wonder if ystematic hyperparameter optimization (e.g., using RandomizedSearchCV or KerasTuner) for learning rates, layer sizes, dropout rates, and time_steps could improved performance, especially for the underperforming LSTM and CNN.

Our model did not include simple baseline model (e.g., predicting yesterday's price). Adding a naive baseline would  contextualize the deep learning models' performance gains. Furthermore, model explainability techniques (like analyzing feature importance for the MLP or attention weights if used) would  provided insight into what the models were basing their predictions on.

This analysis focused on aggregate metrics.Looking into errors—plotting residuals over time, identifying periods of high volatility where models failed (e.g., market crashes), or clustering mispredictions—could reveal specific weaknesses and guide model improvement.

 Refactoring code into functions and creating a clear pipeline would enhance reproducibility. Also, the training warnings about TensorFlow retracing (related to tf.function) suggest opportunities to optimize the graphs formation for better training speed (as seen in TensorFlow documentation on controlling retracing)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import shap
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def load_data(files):
    data = {}
    for file in files:
        try:
            file.seek(0)
            df = pd.read_csv(BytesIO(file.read()))
            data[file.name] = df
        except Exception as e:
            st.error(f"Error reading file {file.name}: {e}")
    return data

def merge_dataframes(data_dict, time_column):
    merged_df = None
    for filename, df in data_dict.items():
        if time_column and time_column in df.columns:
            try:
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                df.set_index(time_column, inplace=True)
            except Exception as e:
                st.warning(f"Error processing time column '{time_column}' in file {filename}: {e}")
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer', rsuffix=f'_{filename}')
    return merged_df

def preprocess_data(data, features, target, time_column=None):
    data_clean = data.dropna(subset=features + [target])
    X = data_clean[features].copy()
    y = data_clean[target].copy()

    if time_column and time_column in data_clean.columns:
        try:
            data_clean[time_column] = pd.to_datetime(data_clean[time_column], errors='coerce')
            X[time_column] = data_clean[time_column].astype(np.int64) // 10**9
        except Exception as e:
            st.warning(f"Error converting time column '{time_column}': {e}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, scaler

def train_model(X, y, model_type, hyperparams=None):
    if model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42, **hyperparams) if hyperparams else RandomForestRegressor(random_state=42)
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Neural Network":
        model = MLPRegressor(random_state=42, **hyperparams) if hyperparams else MLPRegressor(random_state=42)
    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(random_state=42, **hyperparams) if hyperparams else xgb.XGBRegressor(random_state=42)
    else:
        st.error(f"Unsupported model type: {model_type}")
        return None

    model.fit(X, y)
    return model

def perform_cross_validation(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    return cv_scores

def tune_hyperparameters(X, y, model_type):
    if model_type == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestRegressor(random_state=42)
    elif model_type == "Neural Network":
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        model = MLPRegressor(random_state=42, max_iter=1000)
    elif model_type == "XGBoost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        model = xgb.XGBRegressor(random_state=42)
    else:
        st.error(f"Unsupported model type for hyperparameter tuning: {model_type}")
        return None

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_

def explain_model(model, X, model_type):
    try:
        if model_type in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model, X)
            shap_values = explainer.shap_values(X)
        elif model_type == "Linear Regression":
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        elif model_type == "Neural Network":
            background = X.sample(min(100, len(X)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X, nsamples=100)
        else:
            st.warning(f"SHAP explanation not supported for model type: {model_type}")
            return None, None
        return shap_values, explainer
    except Exception as e:
        st.warning(f"An error occurred during SHAP analysis: {str(e)}")
        return None, None

def generate_text_explanation(model, X, y, model_type, performance_metrics, cv_scores):
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_)

    top_features = X.columns[np.argsort(feature_importance)[-3:]].tolist() if feature_importance is not None else []

    explanation = f"""
    **Model Performance Explanation:**
    
    The **{model_type}** model achieved an **R² score of {performance_metrics['r2']:.2f}** on the test set, 
    indicating that it explains **{performance_metrics['r2']*100:.1f}%** of the variance in the target variable. 
    The root mean squared error (RMSE) is **{performance_metrics['rmse']:.2f}**, which represents the average 
    deviation of predictions from actual values.

    **Cross-validation Results:**
    - **Mean R² score:** {np.mean(cv_scores):.2f}
    - **Standard deviation of R² scores:** {np.std(cv_scores):.2f}

    The cross-validation results provide a more robust estimate of the model's performance
    across different subsets of the data.
    """

    if top_features:
        explanation += f"""
    **Top Influential Features:**
    The most influential features for this model are **{', '.join(top_features)}**. 
    These features have the strongest impact on the model's predictions.
    """

    explanation += f"""
    **Interpreting the Results:**
    - An R² close to 1 indicates a good fit, while values closer to 0 suggest poor predictive power.
    - The RMSE is in the same units as the target variable. Lower values indicate better accuracy.
    - Consistent cross-validation scores suggest the model generalizes well to unseen data.

    **Next Steps:**
    - Consider feature engineering or selection to improve model performance.
    - If the model underperforms, try collecting more data or exploring other algorithms.
    - Use the SHAP plots to understand how each feature influences predictions.
    - For time series data, investigate lagged features or moving averages to capture temporal patterns.
    """

    return explanation

def plot_feature_importance(model, feature_names, model_type):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        st.warning("Feature importances not available for this model.")
        return None

    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig = px.bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h',
        title='Feature Importances',
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    return fig

def plot_time_series(data, time_column, target_column):
    fig = px.line(data, x=time_column, y=target_column, title=f"{target_column} over Time")
    return fig

def export_model(model, scaler):
    model_buffer = BytesIO()
    joblib.dump((model, scaler), model_buffer)
    model_buffer.seek(0)
    return model_buffer

def compare_models(X, y, models_to_compare):
    results = []
    for name, model in models_to_compare.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        results.append({
            'Model': name,
            'Mean R²': np.mean(cv_scores),
            'Std R²': np.std(cv_scores)
        })
    return pd.DataFrame(results)

def engineer_features(data, features):
    engineered_data = data.copy()
    for feature in features:
        engineered_data[f'{feature}_squared'] = data[feature] ** 2
        engineered_data[f'{feature}_log'] = np.log1p(data[feature])
    return engineered_data

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual Plot")
    return fig

def interpret_linear_model(model, feature_names):
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
    return coef_df

def create_lag_features(data, target, lags):
    for lag in lags:
        data[f'{target}_lag_{lag}'] = data[target].shift(lag)
    return data

def create_rolling_features(data, target, windows):
    for window in windows:
        data[f'{target}_rolling_mean_{window}'] = data[target].rolling(window=window).mean()
    return data
def main():
    st.set_page_config(layout="wide")
    st.title("Air Quality Sensor Calibration Tool")

    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        data_dict = load_data(uploaded_files)

        if data_dict:
            all_columns = list(set().union(*[df.columns for df in data_dict.values()]))
            time_column = st.selectbox("Select time column (optional)", ["None"] + all_columns)
            time_column = None if time_column == "None" else time_column

            merged_data = merge_dataframes(data_dict, time_column)
            st.write("### Merged Data Preview:", merged_data.head())

            features = st.multiselect("Select features for calibration", merged_data.columns.tolist())
            target = st.selectbox("Select target variable", merged_data.columns.tolist())

            if time_column:
                use_lag_features = st.checkbox("Use lag features")
                use_rolling_features = st.checkbox("Use rolling average features")
                
                if use_lag_features:
                    lag_values = st.multiselect("Select lag values", options=[1, 3, 6, 12, 24], default=[1])
                    merged_data = create_lag_features(merged_data, target, lag_values)
                
                if use_rolling_features:
                    window_sizes = st.multiselect("Select rolling window sizes", options=[3, 6, 12, 24], default=[3])
                    merged_data = create_rolling_features(merged_data, target, window_sizes)

                features = [col for col in merged_data.columns if col != target and col != time_column]
                st.write("Updated feature list:", features)

            use_feature_engineering = st.checkbox("Use feature engineering")
            
            model_type = st.selectbox("Select Calibration Algorithm", 
                                      ["Random Forest", "Linear Regression", "Neural Network", "XGBoost"])

            if not features or not target:
                st.error("Please select features and target variable.")
                return

            if target in features:
                st.error("Target variable should not be included in the features list.")
                return

            # Preprocess data
            X, y, scaler = preprocess_data(merged_data, features, target, time_column)

            if use_feature_engineering:
                X = engineer_features(X, features)
                st.write("Features after engineering:", X.columns.tolist())

            # Handle NaN values
            X = X.dropna()
            y = y[X.index]

            if X.shape[0] == 0:
                st.error("After removing rows with NaN values, no data remains. Please check your data quality.")
                return

            st.write(f"**Number of samples after preprocessing:** {X.shape[0]}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if st.button("Train Calibration Model"):
                if model_type != "Linear Regression":
                    with st.spinner("Tuning hyperparameters..."):
                        best_params = tune_hyperparameters(X_train, y_train, model_type)
                    if best_params:
                        st.write("**Best Hyperparameters:**", best_params)
                        model = train_model(X_train, y_train, model_type, best_params)
                    else:
                        model = train_model(X_train, y_train, model_type)
                else:
                    model = train_model(X_train, y_train, model_type)

                if model is None:
                    st.error("Model training failed. Please check the model type and parameters.")
                    return

                # Model Performance
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                performance_metrics = {
                    'r2': test_score,
                    'rmse': rmse
                }

                cv_scores = perform_cross_validation(model, X, y)

                st.subheader("**Model Performance**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train R² Score", f"{train_score:.4f}")
                    st.metric("Test R² Score", f"{test_score:.4f}")
                with col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                    st.metric("Mean CV R² Score", f"{np.mean(cv_scores):.4f}")
                st.write("**Cross-validation R² Scores:**", cv_scores)

                # Scatter plot of predicted vs actual values
                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, opacity=0.6)
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                         mode='lines', name='Ideal Prediction', line=dict(color='red', dash='dash')))
                fig.update_layout(title="Predicted vs Actual Values")
                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance
                fi_fig = plot_feature_importance(model, X.columns, model_type)
                if fi_fig:
                    st.subheader("**Feature Importance**")
                    st.plotly_chart(fi_fig, use_container_width=True)
                else:
                    st.warning("Feature importance not available for this model.")

                # Time Series Plot
                if time_column:
                    ts_fig = plot_time_series(merged_data.reset_index(), time_column, target)
                    st.subheader("**Time Series Plot**")
                    st.plotly_chart(ts_fig, use_container_width=True)

                # SHAP Analysis
                with st.spinner("Performing SHAP analysis..."):
                    shap_values, explainer = explain_model(model, X_test, model_type)

                if shap_values is not None:
                    st.subheader("**SHAP Summary Plot (Bar)**")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate SHAP summary plot: {e}")

                    st.subheader("**SHAP Summary Plot (Beeswarm)**")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate SHAP beeswarm plot: {e}")
                else:
                    st.warning("SHAP analysis could not be performed for this model.")

                # Residual Analysis
                residual_fig = plot_residuals(y_test, y_pred)
                st.subheader("Residual Analysis")
                st.plotly_chart(residual_fig, use_container_width=True)

                # Linear Model Interpretation
                if model_type == "Linear Regression":
                    st.subheader("Linear Model Interpretation")
                    coef_interpretation = interpret_linear_model(model, X.columns)
                    st.dataframe(coef_interpretation)

                # Generate and display text explanation
                st.subheader("**Model Explanation**")
                explanation = generate_text_explanation(model, X, y, model_type, performance_metrics, cv_scores)
                st.markdown(explanation)

                # Export Model
                model_buffer = export_model(model, scaler)
                st.download_button(
                    label="Download Trained Model",
                    data=model_buffer,
                    file_name="calibration_model.joblib",
                    mime="application/octet-stream"
                )

            # Model Comparison
            st.subheader("Model Comparison")
            if st.button("Compare Models"):
                models_to_compare = {
                    'Random Forest': RandomForestRegressor(random_state=42),
                    'Linear Regression': LinearRegression(),
                    'Neural Network': MLPRegressor(random_state=42, max_iter=1000),
                    'XGBoost': xgb.XGBRegressor(random_state=42)
                }
                comparison_results = compare_models(X, y, models_to_compare)
                st.dataframe(comparison_results)

if __name__ == "__main__":
    main()
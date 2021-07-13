import streamlit as st
import pandas as pd
import base64
from PIL import Image
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Calculates performance metrics
def calc_metrics(input_data):

    X = input_data.iloc[:,0]
    y = input_data.iloc[:,1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)  
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    
    explained_variance = explained_variance_score(y_test,y_pred)
    max_error = max_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    msle = mean_squared_log_error(y_test,y_pred)
    medae = median_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    mpd = mean_poisson_deviance(y_test,y_pred)
    mgd = mean_gamma_deviance(y_test,y_pred)
    mape = mean_absolute_percentage_error(y_test,y_pred)

    explained_variance_series = pd.Series(explained_variance, name='Explained_variance')
    max_error_series = pd.Series(max_error, name='Max_Error')
    mae_series = pd.Series(mae, name='Mean_Absolute_Error')
    mse_series = pd.Series(mse, name='Mean_Squared_Error')
    rmse_series = pd.Series(rmse, name='Root_Mean_Squared_Error')
    msle_series = pd.Series(msle, name='Mean_Squared_Log_Error')
    medae_series = pd.Series(medae, name='Median_Absolute_Error')
    r2_series = pd.Series(r2, name='R2_Score')
    mpd_series = pd.Series(mpd, name='Mean_Poisson_Deviance')
    mgd_series = pd.Series(mgd, name='Mean_Gamma_Deviance')
    mape_series = pd.Series(mape, name='Mean_Absolute_Percentage_Error')

    df = pd.concat([explained_variance_series, max_error_series, mae_series, mse_series, rmse_series, msle_series, medae_series, r2_series, mpd_series, mgd_series, mape_series], axis=1)
    return df

# Load example data
def load_example_data():
    df = pd.read_csv('sample_data.csv')
    return df

# Download performance metrics
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">Download CSV File</a>'
    return href

# Sidebar - Header
st.sidebar.header('Input panel')
st.sidebar.markdown("""
[Example CSV file](https://raw.githubusercontent.com/chinya07/regression_model_performance_app/main/sample_data.csv)
""")

# Sidebar panel - Upload input file
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])

# Sidebar panel - Performance metrics
performance_metrics = ['Explained_variance', 'Max_Error', 'MAE', 'MSE','RMSE', 'MSLE', 'MedAE', 'R2', 'MPD', 'MGD', 'MAPE']
selected_metrics = st.sidebar.multiselect('Performance metrics', performance_metrics, performance_metrics)

# Main panel
image = Image.open('logo.png')
st.image(image, width = 500)
st.title('Model Performance Calculator App')
st.markdown("""
This app calculates the model performance metrics given the actual and predicted values.
* **Python libraries:** `base64`, `pandas`, `streamlit`, `scikit-learn`
""")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    confusion_matrix_df = calc_confusion_matrix(input_df)
    metrics_df = calc_metrics(input_df)
    selected_metrics_df = metrics_df[ selected_metrics ]
    st.header('Input data')
    st.write(input_df)
    st.header('Performance metrics')
    st.write(selected_metrics_df)
    st.markdown(filedownload(selected_metrics_df), unsafe_allow_html=True)
else:
    st.info('Awaiting the upload of the input file.')
    if st.button('Use Example Data'):
        input_df = load_example_data()
        confusion_matrix_df = calc_confusion_matrix(input_df)
        metrics_df = calc_metrics(input_df)
        selected_metrics_df = metrics_df[ selected_metrics ]
        st.header('Input data')
        st.write(input_df)
        st.header('Performance metrics')
        st.write(selected_metrics_df)
        st.markdown(filedownload(selected_metrics_df), unsafe_allow_html=True)

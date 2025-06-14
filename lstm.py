import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import product
import warnings
import streamlit as st
from PIL import Image
from statsmodels.tools.eval_measures import rmse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tools.eval_measures import rmse

# Ignore warnings
warnings.filterwarnings("ignore")

# Read dataset and pre-process:
df = pd.read_csv('cleaned_integrated_stock_data.csv')
if df['date'].dtype == 'object':
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['Current price'] = pd.to_numeric(df['Current price'], errors='coerce')
df = df.sort_values(by='date').reset_index(drop=True)

# Streamlit configuration
st.set_page_config(layout="wide")
st.markdown("<style>.main {padding-top: 0px;}</style>", unsafe_allow_html=True)

# Add images
st.sidebar.image("Pic1.png")
st.image("Pic2.png")

# Add main title
st.markdown("<h1 style='text-align: center; margin-top: -20px;'>LSTM Forecasting Model</h1>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Model Parameters")

symbol_list = sorted(df['Name'].unique().tolist())
crypto_symbol = st.sidebar.selectbox("Select stock name or crypto:", symbol_list)

prediction_ahead = st.sidebar.number_input("Prediction Days Ahead", min_value=1, max_value=30, value=15, step=1)

if st.sidebar.button("Predict"):

    # Step 1: Pull crypto data
    filtered_df = df[df['Name'] == crypto_symbol][['date', 'Current price']].reset_index(drop=True)
    filtered_df = filtered_df.set_index('date')

    # prepare train-test split (80% train, 20% test)
    len_test = 10
    train = filtered_df.iloc[:len(filtered_df)-len_test]
    test = filtered_df.iloc[len(filtered_df)-len_test:]

    # Step 2: LSTM model building
    model = Sequential()
    model.add(Input(shape=(7, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 3: Process data for training
    def create_dataset(series, time_step=1):
        X, y = [], []
        for i in range(len(series) - time_step):
            X.append(series[i:(i + time_step), 0])
            y.append(series[i + time_step, 0])
        return np.array(X), np.array(y)

    scaler = MinMaxScaler(feature_range=(-3, 3))
    series = train['Current price'].values
    scaled_series = scaler.fit_transform(series.reshape(-1, 1))

    X_train, y_train = create_dataset(scaled_series, 7)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Step 4: Train LSTM model
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)

    # find the time_step that best fit
    time_steps = range(2, 15)
    min_rmse = 1e9
    best_time_step = 0

    for t in time_steps:
        series = np.insert(test['Current price'].values, 0, train['Current price'].values[-t:])
        scaled_series = scaler.fit_transform(series.reshape(-1, 1))
        X_test, y_test = create_dataset(scaled_series.reshape(-1, 1), t)

        new_sample = scaled_series[-t:].reshape(1, t)
        X_test = np.concatenate((X_test, new_sample), axis=0)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        forecast = model.predict(X_test, verbose=0)

        new_index = test['Current price'].index.append(
            pd.DatetimeIndex([test['Current price'].index[-1] + pd.Timedelta(days=1)])
        )
        prepended_value = pd.Series([train['Current price'].iloc[-1]], index=[train.index[-1]])
        test_series = pd.concat([prepended_value, test['Current price']])

        forecast = scaler.inverse_transform(forecast)
        forecast_df = pd.DataFrame({'Forecast': forecast.flatten()}, new_index - pd.Timedelta(days=1))

        rmse_value = rmse(test_series, forecast_df['Forecast'])
        if rmse_value < min_rmse:
            min_rmse = rmse_value
            best_time_step = t

    # Step 5: Forecast future values
    series = np.insert(test['Current price'].values, 0, train['Current price'].values[-best_time_step:])
    scaled_series = scaler.fit_transform(series.reshape(-1, 1))
    X_test, y_test = create_dataset(scaled_series.reshape(-1, 1), best_time_step)

    new_sample = scaled_series[-best_time_step:].reshape(1, best_time_step)
    X_test = np.concatenate((X_test, new_sample), axis=0)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    forecast = model.predict(X_test, verbose=0)

    new_index = test['Current price'].index.append(
        pd.DatetimeIndex([test['Current price'].index[-1] + pd.Timedelta(days=1)])
    )
    prepended_value = pd.Series([train['Current price'].iloc[-1]], index=[train.index[-1]])
    test_series = pd.concat([prepended_value, test['Current price']])

    forecast = scaler.inverse_transform(forecast)
    forecast[0] = train['Current price'][-1]
    forecast_df = pd.DataFrame({'Forecast': forecast.flatten()}, new_index - pd.Timedelta(days=1))

    series = np.insert(forecast_df['Forecast'].values, 0, train['Current price'].values[-best_time_step:-1])
    scaled_series = scaler.fit_transform(series.reshape(-1, 1))
    for i in range(prediction_ahead):
        X_pred = scaled_series
        X_pred = X_pred.reshape(1, len(scaled_series), 1)
        y_pred = model.predict(X_pred, verbose=0)
        scaled_series = np.append(scaled_series, y_pred)

    series_pred = scaler.inverse_transform(scaled_series.reshape(-1,1))
    date_range = pd.date_range(start=train['Current price'].iloc[-best_time_step:-1].index[0],
                            periods=len(series_pred), freq='D')
    df_pred = pd.DataFrame({'Prediction': series_pred.ravel()}, index=date_range)
    df_pred = df_pred[df_pred.index >= forecast_df['Forecast'].index[0]]

    # Centered layout for metrics
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-around;">
                <div style="background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center;">
                    <h3>Latest Close Price</h3>
                    <p style="font-size: 20px;">${test['Current price'][-1]:,.2f}</p>
                </div>
                <div style="background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center;">
                    <h3>Price After {prediction_ahead} Days</h3>
                    <p style="font-size: 20px;">${df_pred['Prediction'][-1]:,.2f}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Step 6: Plot the results
    fig = go.Figure()

    # Dữ liệu train
    fig.add_trace(go.Scatter(
        x=train.index, y=train['Current price'],
        mode='lines',
        name='Train',
        line=dict(color='blue')
    ))

    # Dữ liệu test
    fig.add_trace(go.Scatter(
        x=test_series.index, y=test_series,
        mode='lines',
        name='Test',
        line=dict(color='green')
    ))

    # Dự đoán
    fig.add_trace(go.Scatter(
        x=df_pred.index, y=df_pred['Prediction'],
        mode='lines',
        name='Predict',
        line=dict(color='orange')
    ))

    # Vạch chia Train/Test
    split1_date = train.index[-1]
    fig.add_shape(
        type='line',
        x0=split1_date, x1=split1_date,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='red', dash='dash')
    )
    fig.add_annotation(
        x=split1_date,
        y=1,
        xref='x', yref='paper',
        text='Train/Test split',
        showarrow=False,
        yanchor='bottom',
        font=dict(color='red'),
        textangle=45
    )

    # Vạch chia Test/Future
    split2_date = test.index[-1]
    fig.add_shape(
        type='line',
        x0=split2_date, x1=split2_date,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='gray', dash='solid')
    )
    fig.add_annotation(
        x=split2_date,
        y=1,
        xref='x', yref='paper',
        text='Test/Future split',
        showarrow=False,
        yanchor='bottom',
        font=dict(color='gray'),
        textangle=45
    )

    # Tùy chỉnh layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        # paper_bgcolor='lightblue',     # Nền toàn vùng figure
        plot_bgcolor='#F0F8FF',       # Nền vùng biểu đồ
        legend=dict(
            x=-0.2,
            y=0,
            xanchor='left',
            yanchor='bottom',
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='Black'
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        width=1200,
        height=600
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)

# Streamlit run lstm.py
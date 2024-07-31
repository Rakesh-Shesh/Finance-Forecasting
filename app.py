import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sklearn.linear_model
import pandas as pd
import streamlit as st
from statistics import mean, median, stdev, variance, StatisticsError
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_data(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    return df


def calculate_error(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100


def parse_month_column(data):
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    data['Month'] = data['Month'].map(month_map)
    data['Month'] = pd.to_datetime(data['Month'], format='%m').dt.strftime('2023-%m-%d')
    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')
    return data

def home_page():
    st.title('Cost Breakdown Analysis')

    # File uploader to upload Excel file
    
    uploaded_file_K1 = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"], key="home_file_uploader")

    if uploaded_file is not None:
        # Read the Excel file to get the sheet names
        sheet_names = pd.ExcelFile(uploaded_file_K1).sheet_names

        # Dropdown to select sheet
        selected_sheet = st.selectbox('Select Sheet:', sheet_names, key="sheet_selectbox_1")

        # Read the data from the selected sheet
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

        # Remove any commas from numeric values and convert to integers
        df = df.replace({',': ''}, regex=True)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

        # List of months
        months = df.columns[1:]

        # Melt the dataframe for easier plotting
        df_melted = df.melt(id_vars=['Category'], value_vars=months, var_name='Month', value_name='Cost')

        # Pie Chart
        st.header('Pie Chart')
        selected_month_pie = st.selectbox('Select Month for Pie Chart:', months, key="pie_month_selectbox_1")
        filtered_df_pie = df[['Category', selected_month_pie]].rename(columns={selected_month_pie: 'Cost'})
        fig_pie = px.pie(filtered_df_pie, names='Category', values='Cost',
                         title=f'Cost Breakdown for {selected_month_pie}')
        st.plotly_chart(fig_pie)

        # Bar Chart
        st.header('Bar Chart')
        selected_month_bar = st.selectbox('Select Month for Bar Chart:', months, index=1, key="bar_month_selectbox_1")
        filtered_df_bar = df[['Category', selected_month_bar]].rename(columns={selected_month_bar: 'Cost'})
        fig_bar = px.bar(filtered_df_bar, x='Category', y='Cost',
                         title=f'Cost Breakdown for {selected_month_bar}',
                         labels={'Category': 'Category', 'Cost': 'Cost'})
        st.plotly_chart(fig_bar)

        # Stacked Column Chart
        st.header('Stacked Column Chart')
        fig_stacked = go.Figure()
        for category in df['Category']:
            fig_stacked.add_trace(go.Bar(
                x=months,
                y=df[df['Category'] == category].iloc[0, 1:],
                name=category
            ))

        fig_stacked.update_layout(
            barmode='stack',
            title='Monthly Cost Breakdown',
            xaxis_title='Month',
            yaxis_title='Cost'
        )
        st.plotly_chart(fig_stacked)

if __name__ == '__main__':
    home_page()
def descriptive_statistics_page():
    # Function to preprocess the data
    def preprocess_data(df):
        df['Month'] = pd.to_datetime(df['Month'], format='%d-%m-%Y')
        df['Year'] = df['Month'].dt.year
        df['Month_name'] = df['Month'].dt.strftime('%b')  # Get month name
        return df

    # Function to format y-axis values in millions
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)

    # Function to plot YoY comparison with bar chart
    def plot_yoy_comparison_bar(df, cost_head):
        monthly_data = df.groupby(['Year', df['Month'].dt.month])[cost_head].sum().unstack(level=0)
        monthly_data.index = df['Month'].dt.strftime('%b').unique()  # Set index as month names

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the bar chart
        monthly_data.plot(kind='bar', ax=ax)

        # Format y-axis
        formatter = FuncFormatter(millions)
        ax.yaxis.set_major_formatter(formatter)

        plt.title(f'YoY Comparison of {cost_head} (Bar Chart)')
        plt.xlabel('Month')
        plt.ylabel('Cost (in millions)')
        plt.legend(title='Year')
        st.pyplot(fig)

    # Function to plot YoY comparison with line chart
    def plot_yoy_comparison_line(df, cost_head):
        monthly_data = df.groupby(['Year', df['Month'].dt.month])[cost_head].sum().unstack(level=0)
        monthly_data.index = df['Month'].dt.strftime('%b').unique()  # Set index as month names

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the line chart
        for year in monthly_data.columns:
            ax.plot(monthly_data.index, monthly_data[year], marker='o', label=str(year))

        # Format y-axis
        formatter = FuncFormatter(millions)
        ax.yaxis.set_major_formatter(formatter)

        plt.title(f'YoY Comparison of {cost_head} (Line Chart)')
        plt.xlabel('Month')
        plt.ylabel('Cost (in millions)')
        plt.legend(title='Year')
        st.pyplot(fig)

    # Function to plot grouped bar chart for cost heads
    def plot_grouped_bar_cost_heads(df):
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cost_heads = [col for col in numeric_cols if col not in ['Year', 'Month']]

        cost_head_totals = df.groupby('Year')[cost_heads].sum()
        cost_head_percentage = cost_head_totals.div(cost_head_totals.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        cost_head_percentage.T.plot(kind='bar', ax=ax)

        plt.title('Percentage of Total Cost by Cost Head')
        plt.xlabel('Cost Head')
        plt.ylabel('Percentage of Total Cost (%)')
        plt.legend(title='Year')
        st.pyplot(fig)

    # Streamlit app
    st.title('YoY Comparison of Cost Heads')
    st.write("Upload your dataset in Excel or CSV format.")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"],key="file_uploader_2")

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        df = preprocess_data(df)
        cost_heads = [col for col in df.columns if col not in ['Month', 'Year', 'Month_name']]

        # Plot the grouped bar chart for cost heads
        st.header(f"Total Outflow/Inflow by Category (YoY)", divider='orange')
        plot_grouped_bar_cost_heads(df)
        st.header(f'Category by Month (YoY)', divider='orange')
        # Select cost head
        cost_head = st.selectbox('Select Cost Head', cost_heads)

        # Plot the data
        st.write("### Bar Chart")
        plot_yoy_comparison_bar(df, cost_head)

        st.write("### Line Chart")
        plot_yoy_comparison_line(df, cost_head)
    else:
        st.write("Please upload a file to proceed.")

    def load_data(file, sheet_name):
        data = pd.read_excel(file, sheet_name=sheet_name)
        return data

    # Function to calculate statistics
    def calculate_statistics(data):
        stats = {}
        try:
            stats['mean'] = mean(data)
        except StatisticsError:
            stats['mean'] = None

        try:
            stats['median'] = median(data)
        except StatisticsError:
            stats['median'] = None

        try:
            stats['stdev'] = stdev(data)
        except StatisticsError:
            stats['stdev'] = None

        try:
            stats['variance'] = variance(data)
        except StatisticsError:
            stats['variance'] = None

        try:
            stats['cv'] = (stats['stdev'] / stats['mean']) if stats['mean'] != 0 else None
        except TypeError:
            stats['cv'] = None

        # Calculating outliers using IQR method
        q1 = pd.Series(data).quantile(0.25)
        q3 = pd.Series(data).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        stats['outliers'] = [x for x in data if x < lower_bound or x > upper_bound]

        return stats

    # Function to get outliers for a specific year or cashflow type
    def get_combined_outliers(data_list):
        combined_data = [item for sublist in data_list for item in sublist]
        combined_stats = calculate_statistics(combined_data)
        return combined_stats['outliers']

    # Streamlit app
    def main():
        st.title("Descriptive Statistics")

        # File uploader
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx",key="file_uploader_3")
        if uploaded_file is not None:
            # Year selection
            year = st.selectbox("Select Year", ["2022", "2023"])

            # Cashflow type selection
            cashflow_types = st.multiselect("Select Cashflow Type", ["Inflow", "Outflow"])

            sheets_to_load = []
            metrics_data_list = []
            for cashflow in cashflow_types:
                if year == "2022":
                    sheets_to_load.append(f"2022-{cashflow.lower()}")
                elif year == "2023":
                    sheets_to_load.append(f"2023-{cashflow.lower()}")

            if sheets_to_load:
                for sheet in sheets_to_load:
                    st.write(f"### Data from sheet: {sheet}")
                    df = load_data(uploaded_file, sheet)

                    # Display dataframe
                    st.write("## Data")
                    st.write(df)

                    # Select metric column
                    metric_columns = df.columns[1:]  # Assuming the first column is 'Month' and the rest are metrics
                    selected_metric = st.selectbox(f"Select Metric for {sheet}", metric_columns)

                    if selected_metric:
                        metrics_data = df[selected_metric].dropna().tolist()
                        metrics_data_list.append(metrics_data)

                        stats = calculate_statistics(metrics_data)

                        st.write(f"## Statistics for {selected_metric} in {sheet}")
                        st.write(f"**Mean**: {stats['mean']}")
                        st.write(f"**Median**: {stats['median']}")
                        st.write(f"**Standard Deviation**: {stats['stdev']}")
                        st.write(f"**Variance**: {stats['variance']}")
                        st.write(f"**Coefficient of Variation**: {stats['cv']}")
                        st.write(f"**Outliers**: {stats['outliers']}")

                # Display outliers for the selected year
                year_outliers = get_combined_outliers(metrics_data_list)

                # Display outliers for each cashflow type
                for cashflow in cashflow_types:
                    cashflow_outliers = []
                    for sheet in sheets_to_load:
                        if cashflow.lower() in sheet:
                            df = load_data(uploaded_file, sheet)
                            metrics_data = df[selected_metric].dropna().tolist()
                            cashflow_outliers.extend(metrics_data)
                    cashflow_outliers = get_combined_outliers([cashflow_outliers])

    if __name__ == "__main__":
        main()


def Trend_Insights_page():
    # Set the title of the Streamlit app
    st.title("Excel File Analytics with Line Chart")

    # Upload the Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"],key="file_uploader_4")

    if uploaded_file:
        # Load the Excel file
        excel_data = pd.ExcelFile(uploaded_file)

        # Load each sheet into a dictionary
        sheets_dict = {sheet_name: excel_data.parse(sheet_name) for sheet_name in excel_data.sheet_names}

        # Display the sheet names
        st.write("Sheets available in the Excel file:", list(sheets_dict.keys()))

        # Multiselect for columns from each sheet
        selected_columns = []
        for sheet_name, df in sheets_dict.items():
            columns = df.columns
            selected = st.multiselect(f"Select columns from {sheet_name}", columns)
            selected_columns.extend([(sheet_name, col) for col in selected])

        if selected_columns:
            # List to store data for plotting
            plot_data = {}

            # Iterate over the selected columns and extract data
            for (sheet_name, col) in selected_columns:
                if 'Month' not in sheets_dict[sheet_name].columns:
                    st.error(f"'Month' column not found in {sheet_name}")
                    continue
                plot_data[f"{sheet_name} - {col}"] = sheets_dict[sheet_name].set_index('Month')[col]

            # Create a DataFrame for plotting
            plot_df = pd.DataFrame(plot_data)

            # Normalize the data to show trends
            scaler = StandardScaler()
            plot_df_normalized = pd.DataFrame(scaler.fit_transform(plot_df), index=plot_df.index,
                                              columns=plot_df.columns)

            # Plot the line chart with normalized data using Plotly
            fig = px.line(plot_df_normalized, title='Trend Analysis',
                          labels={'value': 'Normalized Value', 'index': 'Month'})
            fig.update_layout(xaxis_title='Month', yaxis_title='Normalized Value')
            st.plotly_chart(fig)

            # Option to download the normalized plot data
            st.download_button(
                label="Download normalized data as CSV",
                data=plot_df_normalized.to_csv().encode('utf-8'),
                file_name='normalized_plot_data.csv',
                mime='text/csv'
            )


# Page navigation
page = st.sidebar.radio("Select a page",
                        ["Home", "Descriptive Statistics", "Trend Insights", "Correlation", "Prediction",
                         "Simulation"])


def correlation_page():
    st.title("Correlation Matrix - Cashflow Prediction")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"],key="file_uploader_5")

    if uploaded_file is not None:
        try:
            # Read the sheet names
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names

            # Check for the required sheets
            if "Passenger Aircraft" not in sheet_names or "Freight Aircraft" not in sheet_names:
                st.error("The Excel file must contain 'Passenger Aircraft' and 'Freight Aircraft' sheets.")
            else:
                # Select the type of aircraft
                aircraft_type = st.selectbox("Select Aircraft Type", ["Passenger Aircraft", "Freight Aircraft"])

                # Read the selected sheet
                df = pd.read_excel(uploaded_file, sheet_name=aircraft_type)

                st.write(f"DataFrame ({aircraft_type}):")
                st.write(df)

                # Select only numeric columns
                numeric_df = df.select_dtypes(include=['number'])

                if numeric_df.empty:
                    st.error("The uploaded file does not contain any numeric data.")
                else:
                    # Calculate the correlation matrix
                    corr_matrix = numeric_df.corr()

                    st.write("Correlation Matrix:")
                    st.write(corr_matrix)

                    # Handle large number of columns by allowing the user to select a subset of columns
                    columns_to_display = st.multiselect(
                        'Select columns to display in correlation matrix',
                        numeric_df.columns,
                        default=list(numeric_df.columns[:10])  # Convert to list for default selection
                    )

                    if len(columns_to_display) > 0:
                        subset_corr_matrix = corr_matrix.loc[columns_to_display, columns_to_display]

                        # Plot the correlation matrix
                        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
                        sns.heatmap(subset_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Please select at least one column to display the correlation matrix.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


def ml_algorithms_page():


    st.title('G10X Forecasting Engine')

    # File uploader for CSV or Excel
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"],key="file_uploader_6")

    if uploaded_file is not None:
        # Determine file type and load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.header("Uploaded Data",divider='orange')
        st.dataframe(df.head())

        # Exclude 'month' column and prepare columns list for target selection
        columns = [col for col in df.columns if col != 'month']
        
        # User inputs for columns and parameters
        timestamp_column = st.selectbox("Timestamp Column Name", columns)
        target_columns = st.multiselect("Target Column Names", columns)
        prediction_length = st.number_input("Prediction Length", min_value=1, value=48)
        eval_metric = st.selectbox("Evaluation Metric", ["MASE", "MAPE", "RMSE"], index=0)
        presets123 = st.selectbox("Presets", ["medium_quality", "high_quality"], index=0)
        time_lim = st.number_input("Time Limit (seconds) per model", min_value=1, value=600)

        # Create an ID column if it doesn't exist
        df['item_id'] = 0

        # Ensure timestamp column is in datetime format and sort the data
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(by=[timestamp_column])

        # Prepare data
        train_data = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column='item_id',
            timestamp_column=timestamp_column
        )
        st.header("Train Data",divider='orange')
        st.dataframe(train_data.head())

        # Split data into train and test sets
        train_data, test_data = train_data.train_test_split(prediction_length=prediction_length)

        # Loop through each selected target column and train a model
        for target_column in target_columns:
            st.write(f"### Running model for target column: {target_column}")
            
            # Initialize predictor
            predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                target=target_column,
                eval_metric=eval_metric,
                quantile_levels=[0.05, 0.5, 0.95]
            )

            # Fit predictor
            with st.spinner(f"Training the model for {target_column}..."):
                predictor.fit(
                    train_data,
                    presets=presets123,
                    time_limit=time_lim
                )

            # Predict
            predictions = predictor.predict(test_data)
            st.header(f"Predictions for {target_column}",divider='orange')
            st.dataframe(predictions.head())

            # Interpretation of predictions
            st.header("Interpretation of Predictions")
            st.markdown("Quantile forecast represents the quantiles of the forecast distribution. For example, if the 0.1 quantile (also known as P10, or the 10th percentile) is equal to x, it means that the time series value is predicted to be below x 10% of the time. As another example, the 0.5 quantile (P50) corresponds to the median forecast. Quantiles can be used to reason about the range of possible outcomes. For instance, by the definition of the quantiles, the time series is predicted to be between the P10 and P90 values with 80% probability.")

            st.pyplot(predictor.plot(test_data, predictions, quantile_levels=[0.05, 0.95], max_history_length=200, max_num_item_ids=4))

            # Display leaderboard
            leaderboard = predictor.leaderboard(test_data)
            st.header(f"Leaderboard for {target_column}",divider='orange')
            st.dataframe(leaderboard)


def Simulation_Page():
    # Function to load data from CSV or Excel
    def load_data(uploaded_file):
        if uploaded_file.type == "text/csv":
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        data['Month'] = pd.to_datetime(data['Month'], format='%d-%m-%Y')
        return data

    # Function to prepare data for Prophet
    def prepare_prophet_data(df, column_name):
        df_prophet = df[['Month', column_name]].rename(columns={'Month': 'ds', column_name: 'y'})
        return df_prophet

    # Function to predict using Prophet and ensure positive values
    def predict_prophet(df_prophet):
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].abs()  # Replace negative values with their positive counterparts
        return forecast[['ds', 'yhat']]

    # Function to calculate MAPE
    def calculate_mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)

    # Streamlit app
    st.title("Cashflow Simulation (2024)")

    st.write("Upload Inflow and Outflow files (CSV or Excel) for 2022 and 2023.")
    uploaded_inflow = st.file_uploader("Upload Inflow File", type=["csv", "xlsx", "xls"],key="file_uploader_7")
    uploaded_outflow = st.file_uploader("Upload Outflow File", type=["csv", "xlsx", "xls"],key="file_uploader_8")

    if uploaded_inflow is not None and uploaded_outflow is not None:
        # Load data
        inflow_df = load_data(uploaded_inflow)
        outflow_df = load_data(uploaded_outflow)

        st.header("Inflow Data (Head)",divider='orange')
        st.dataframe(inflow_df.head())

        st.header("Outflow Data (Head)",divider='orange')
        st.dataframe(outflow_df.head())

        # Predict for each cost head in inflow and outflow
        inflow_predictions = pd.DataFrame()
        outflow_predictions = pd.DataFrame()
        inflow_predictions['Month'] = pd.date_range(start='2024-01-01', periods=12, freq='M')
        outflow_predictions['Month'] = pd.date_range(start='2024-01-01', periods=12, freq='M')

        mape_inflow = {}
        mape_outflow = {}

        for column in inflow_df.columns[1:]:
            df_prophet = prepare_prophet_data(inflow_df[['Month', column]], column)
            forecast = predict_prophet(df_prophet)
            inflow_predictions[column] = forecast['yhat'].tail(12).values
            mape_inflow[column] = calculate_mape(inflow_df[column].values, forecast['yhat'][:len(inflow_df)].values)

        for column in outflow_df.columns[1:]:
            df_prophet = prepare_prophet_data(outflow_df[['Month', column]], column)
            forecast = predict_prophet(df_prophet)
            outflow_predictions[column] = forecast['yhat'].tail(12).values
            mape_outflow[column] = calculate_mape(outflow_df[column].values, forecast['yhat'][:len(outflow_df)].values)

        st.header("Inflow Predictions for 2024 (Head)",divider='orange')
        st.dataframe(inflow_predictions.head())

        st.header("Outflow Predictions for 2024 (Head)",divider='orange')
        st.dataframe(outflow_predictions.head())

        st.header("MAPE for Inflow Predictions",divider='orange')
        mape_inflow_df = pd.DataFrame.from_dict(mape_inflow, orient='index', columns=['MAPE']).reset_index()
        mape_inflow_df = mape_inflow_df.rename(columns={'index': 'Cost Head'})
        st.dataframe(mape_inflow_df)

        st.header("MAPE for Outflow Predictions",divider='orange')
        mape_outflow_df = pd.DataFrame.from_dict(mape_outflow, orient='index', columns=['MAPE']).reset_index()
        mape_outflow_df = mape_outflow_df.rename(columns={'index': 'Cost Head'})
        st.dataframe(mape_outflow_df)

        st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)

        # Simulation part
        st.header("Simulation")

        st.subheader("Adjust Inflow")
        inflow_month = st.selectbox("Select Month for Inflow", options=inflow_predictions['Month'].dt.strftime('%B-%Y'))
        inflow_scenario = st.selectbox("Select Scenario for Inflow", options=["Best Scenario", "Worst Scenario", "Realistic Scenario"])

        inflow_selected = inflow_predictions[inflow_predictions['Month'].dt.strftime('%B-%Y') == inflow_month].squeeze()
        inflow_sliders = {}

        for column in inflow_predictions.columns[1:]:
            if mape_inflow[column] < 0.5:
                mape = mape_inflow[column]
                if inflow_scenario == "Best Scenario":
                    default_value = inflow_selected[column] * (1 + mape)
                elif inflow_scenario == "Worst Scenario":
                    default_value = inflow_selected[column] * (1 - mape)
                else:  # Realistic Scenario
                    default_value = inflow_selected[column]
                inflow_sliders[column] = st.slider(f"{column}", min_value=0.0, max_value=default_value * 2, value=default_value)

        total_inflow = sum(inflow_sliders.values())
        st.markdown(f"<h4>Total Cash Gained (Inflow) for {inflow_month} ({inflow_scenario}): {total_inflow:.2f}</h4>", unsafe_allow_html=True)

        st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)

        st.subheader("Adjust Outflow")
        outflow_month = st.selectbox("Select Month for Outflow", options=outflow_predictions['Month'].dt.strftime('%B-%Y'))
        outflow_scenario = st.selectbox("Select Scenario for Outflow", options=["Best Scenario", "Worst Scenario", "Realistic Scenario"])

        outflow_selected = outflow_predictions[outflow_predictions['Month'].dt.strftime('%B-%Y') == outflow_month].squeeze()
        outflow_sliders = {}

        for column in outflow_predictions.columns[1:]:
            if mape_outflow[column] < 0.5:
                mape = mape_outflow[column]
                if outflow_scenario == "Best Scenario":
                    default_value = outflow_selected[column] * (1 - mape)
                elif outflow_scenario == "Worst Scenario":
                    default_value = outflow_selected[column] * (1 + mape)
                else:  # Realistic Scenario
                    default_value = outflow_selected[column]
                outflow_sliders[column] = st.slider(f"{column}", min_value=0.0, max_value=default_value * 2, value=default_value)

        total_outflow = sum(outflow_sliders.values())
        st.markdown(f"<h4>Total Cash Required (Outflow) for {outflow_month} ({outflow_scenario}): {total_outflow:.2f}</h4>", unsafe_allow_html=True)

        st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)
    else:
        st.write("Please upload both inflow and outflow files.")

if page == "Home":
    home_page()
elif page == "Correlation":
    correlation_page()
elif page == "Prediction":
    ml_algorithms_page()
elif page == "Descriptive Statistics":
    descriptive_statistics_page()
elif page == "Trend Insights":
    Trend_Insights_page()
elif page == "Simulation":
    Simulation_Page()
    

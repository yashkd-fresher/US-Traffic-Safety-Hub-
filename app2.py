import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
import subprocess
# Set page configuration
st.set_page_config(
    page_title="US Accident Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .severity-1 {
        background-color: #BFDBFE;
        color: #1E3A8A;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .severity-2 {
        background-color: #93C5FD;
        color: #1E3A8A;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .severity-3 {
        background-color: #60A5FA;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .severity-4 {
        background-color: #2563EB;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----- Helper Functions -----
@st.cache_data
def load_data(dataset_name="sobhanmoosavi/us-accidents"):
    """Load the dataset from Kaggle with caching"""
    try:
        # Check if dataset already exists locally
        file_path = "dataset.csv"
        if not os.path.exists(file_path):
            st.info("Downloading dataset from Kaggle. This may take a moment...")
            
            # Set up Kaggle API credentials if not already set
            if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
                # Create the directory if it doesn't exist
                os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
                
                # Get API credentials from Streamlit secrets or user input
                kaggle_username = st.secrets.get("kaggle_username", "") if hasattr(st, "secrets") else ""
                kaggle_key = st.secrets.get("kaggle_key", "") if hasattr(st, "secrets") else ""
                
                # If not in secrets, ask the user
                if not kaggle_username or not kaggle_key:
                    st.warning("Kaggle API credentials are required to download the dataset.")
                    kaggle_username = st.text_input("Kaggle Username")
                    kaggle_key = st.text_input("Kaggle API Key", type="password")
                    
                    if not kaggle_username or not kaggle_key:
                        st.error("Please provide Kaggle credentials to continue.")
                        return None
                
                # Create kaggle.json file
                with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                    f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')
                
                # Set permissions for the file
                os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
            
            # Download dataset using Kaggle API
            try:
                subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name, '--unzip'], 
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Find the CSV file in the downloaded data
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                if csv_files:
                    # Rename the first CSV to our expected filename
                    os.rename(csv_files[0], file_path)
                    st.success("Dataset successfully downloaded!")
                else:
                    st.error("No CSV file found in the downloaded dataset.")
                    return None
                    
            except subprocess.CalledProcessError as e:
                st.error(f"Error downloading dataset: {e.stderr.decode()}")
                return None
        
        
        
        # Convert time columns to datetime
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
        df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')
        
        # Extract useful features
        df['Date'] = df['Start_Time'].dt.date
        df['Hour'] = df['Start_Time'].dt.hour
        df['Day_of_Week'] = df['Start_Time'].dt.day_name()
        df['Month'] = df['Start_Time'].dt.month_name()
        df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60  # in minutes
        
        # Ensure Weather_Condition is string type
        if 'Weather_Condition' in df.columns:
            df['Weather_Condition'] = df['Weather_Condition'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    

    
@st.cache_resource
def load_model(model_path):
    """Load the pre-trained model with caching"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please check the file path.")
        return None

@st.cache_resource
def load_explainer(explainer_path):
    """Load the SHAP explainer with caching"""
    try:
        with open(explainer_path, 'rb') as file:
            explainer = pickle.load(file)
        return explainer
    except FileNotFoundError:
        st.error(f"Explainer file not found at {explainer_path}. Please check the file path.")
        return None

@st.cache_data
def get_model_features(features_path):
    """Load the model features list with caching"""
    try:
        with open(features_path, 'rb') as file:
            features = pickle.load(file)
        return features
    except FileNotFoundError:
        st.error(f"Features file not found at {features_path}. Please check the file path.")
        return None

def create_accident_map(df, time_range, day_filter, weather_filter):
    """Create an interactive map showing accident locations colored by severity"""
    # Filter data based on user selection
    filtered_df = df.copy()
    
    # In create_accident_map function
    if time_range:
        filtered_df = filtered_df[(filtered_df['Hour'] >= float(time_range[0])) & 
                          (filtered_df['Hour'] <= float(time_range[1]))]
    
    if day_filter:
        filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(day_filter)]
    
    if weather_filter and 'Weather_Condition' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Weather_Condition'].isin(weather_filter)]
    
    # Sample data if there are too many points to display
    sample_size = min(5000, len(filtered_df))
    map_data = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
    
    # Create the map
    fig = px.scatter_mapbox(
        map_data, 
        lat='Start_Lat', 
        lon='Start_Lng',
        color='Severity',
        color_continuous_scale='blues',
        size_max=15,
        zoom=3,
        mapbox_style="carto-positron",
        center={"lat": 37.0902, "lon": -95.7129},  # Center of US
        opacity=0.7,
        hover_name="ID",
        hover_data={
            "Severity": True,
            "Hour": True,
            "Day_of_Week": True,
            "Duration": True,
            "Start_Lat": False,
            "Start_Lng": False
        },
        title="Geographic Distribution of Accidents"
    )
    
    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            title="Severity",
            tickvals=[1, 2, 3, 4],
            ticktext=["1 - Minor", "2 - Moderate", "3 - Significant", "4 - Severe"]
        )
    )
    
    return fig

def create_heatmap_layer(df, time_range, day_filter, weather_filter):
    """Create a density heatmap layer showing accident hotspots"""
    # Filter data based on user selection
    filtered_df = df.copy()
    
    if time_range:
        filtered_df = filtered_df[(filtered_df['Hour'] >= time_range[0]) & 
                                  (filtered_df['Hour'] <= time_range[1])]
    
    if day_filter:
        filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(day_filter)]
    
    if weather_filter and 'Weather_Condition' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Weather_Condition'].isin(weather_filter)]
    
    # Sample data if there are too many points
    sample_size = min(10000, len(filtered_df))
    map_data = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
    
    # Create a density heatmap using Plotly's Densitymapbox
    fig = px.density_mapbox(
        map_data,
        lat='Start_Lat',
        lon='Start_Lng',
        z='Severity',  # Color by severity
        radius=10,
        center={"lat": 37.0902, "lon": -95.7129},
        zoom=3,
        mapbox_style="carto-positron",
        opacity=0.7,
        title="Accident Hotspot Density Map"
    )
    
    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            title="Severity Density"
        )
    )
    
    return fig

def create_prediction_form():
    """Create form for accident severity prediction"""
    st.subheader("Accident Severity Prediction Tool")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.selectbox("Day of Week", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            month = st.selectbox("Month", options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            
        with col2:
            latitude = st.number_input("Latitude", min_value=25.0, max_value=49.0, value=37.7749, step=0.01)
            longitude = st.number_input("Longitude", min_value=-125.0, max_value=-65.0, value=-122.4194, step=0.01)
            distance = st.number_input("Distance (mi)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            
        with col3:
            duration = st.number_input("Estimated Duration (minutes)", min_value=5, max_value=300, value=45, step=5)
            
            # Add weather condition if available in the dataset
            if 'Weather_Condition' in st.session_state.get('weather_options', []):
                weather = st.selectbox("Weather Condition", 
                                      options=st.session_state.weather_options)
                visibility = st.slider("Visibility (miles)", 0.0, 10.0, 5.0, 0.1)
            else:
                weather = "Clear"  # Default value
                visibility = 5.0   # Default value
            
        # Additional inputs if available in the dataset
        if 'Temperature(F)' in st.session_state.get('feature_columns', []):
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.slider("Temperature (°F)", 0, 100, 65, 1)
            with col2:
                humidity = st.slider("Humidity (%)", 0, 100, 50, 1)
            with col3:
                wind_speed = st.slider("Wind Speed (mph)", 0, 50, 10, 1)
        else:
            temperature = 65  # Default value
            humidity = 50     # Default value
            wind_speed = 10   # Default value
            
        submit_button = st.form_submit_button("Predict Severity", use_container_width=True)
    
    return {
        'Hour': hour,
        'Day_of_Week': day,
        'Month': month,
        'Start_Lat': latitude,
        'Start_Lng': longitude,
        'Distance(mi)': distance,
        'Duration': duration,
        'Weather_Condition': weather,
        'Visibility(mi)': visibility,
        'Temperature(F)': temperature,
        'Humidity(%)': humidity,
        'Wind_Speed(mph)': wind_speed,
        'submit': submit_button
    }

def prepare_prediction_data(input_data, features_list):
    """Prepare the input data for prediction"""
    # Create a dataframe with the input data
    input_df = pd.DataFrame([input_data])
    
    # Prepare the features based on the model's expected input
    prediction_data = pd.DataFrame(index=[0], columns=features_list)
    prediction_data = prediction_data.fillna(0)  # Fill with zeros initially
    
    # Fill in the values that we have
    for col in features_list:
        # Handle day of week columns
        if col.startswith('Day_'):
            day_value = input_data['Day_of_Week']
            if col == f'Day_{day_value}':
                prediction_data[col] = 1
                
        # Handle month columns
        elif col.startswith('Month_'):
            month_value = input_data['Month']
            if col == f'Month_{month_value}':
                prediction_data[col] = 1
                
        # Handle weather columns
        elif col.startswith('Weather_'):
            weather_value = input_data['Weather_Condition']
            if col == f'Weather_{weather_value}':
                prediction_data[col] = 1
                
        # Handle direct numerical features
        # In prepare_prediction_data function
        elif col in input_data:
    # Ensure numeric conversion
            try:
              prediction_data[col] = float(input_data[col])
            except (ValueError, TypeError):
               prediction_data[col] = 0  # Default value if conversion fails
    
    return prediction_data

def plot_feature_importance(prediction_data, model, explainer):
    """Plot feature importance for the prediction"""
    # Get SHAP values for the prediction
    shap_values = explainer.shap_values(prediction_data)
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get feature importance
    if isinstance(shap_values, list):  # For multiclass models
        # Sum the absolute SHAP values across all classes
        shap_sum = np.abs(np.array(shap_values)).sum(axis=0)
    else:
        shap_sum = np.abs(shap_values).sum(axis=0)
    
    feature_importance = pd.DataFrame({
        'Feature': prediction_data.columns,
        'SHAP_importance': shap_sum
    })
    
    feature_importance = feature_importance.sort_values('SHAP_importance', ascending=False).head(10)
    
    # Plot the feature importance
    sns.barplot(x='SHAP_importance', y='Feature', data=feature_importance, palette='Blues_d', ax=ax)
    ax.set_title('Top 10 Features by SHAP Importance', fontsize=14)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    return fig

def provide_mitigation_suggestions(prediction, input_data):
    """Provide mitigation suggestions based on the prediction and input data"""
    suggestions = []
    
    # Convert prediction to int if it's a numpy array or other non-int type
    if not isinstance(prediction, int):
        prediction = int(prediction)
    
    if prediction >= 3:  # High severity
        if input_data['Hour'] >= 17 and input_data['Hour'] <= 19:
            suggestions.append("Consider adjusting travel time to avoid rush hour (5-7 PM).")
        
        if input_data.get('Weather_Condition') in ['Rain', 'Snow', 'Fog']:
            suggestions.append(f"Extreme caution advised during {input_data['Weather_Condition']} conditions.")
        
        if input_data.get('Visibility(mi)', 10) < 3:
            suggestions.append("Low visibility conditions. Consider postponing travel if possible.")
        
        if input_data.get('Wind_Speed(mph)', 0) > 20:
            suggestions.append("High wind speeds. Be cautious of large vehicles and potential debris.")
        
        if input_data['Day_of_Week'] in ['Friday', 'Saturday']:
            suggestions.append(f"Higher accident risk on {input_data['Day_of_Week']}s. Consider extra caution.")
    
    elif prediction == 2:  # Medium severity
        if input_data['Hour'] >= 7 and input_data['Hour'] <= 9:
            suggestions.append("Consider adjusting travel time to avoid morning rush hour (7-9 AM).")
        
        if input_data.get('Weather_Condition') in ['Rain', 'Cloudy']:
            suggestions.append(f"Increased caution advised during {input_data['Weather_Condition']} conditions.")
        
        if input_data.get('Visibility(mi)', 10) < 5:
            suggestions.append("Reduced visibility. Maintain safe following distance.")
    
    else:  # Low severity
        suggestions.append("Low risk conditions, but always maintain safe driving practices.")
    
    # Add general suggestions
    suggestions.append("Always wear seatbelts and avoid distracted driving.")
    
    return suggestions

# ----- Main Application -----

def main():
    # Load the CSS
    st.markdown('<h1 class="main-header">US Accident Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car-accident.png", width=80)
        st.title("Configuration")
         
        kaggle_dataset = st.text_input(
            "Kaggle Dataset", 
            "sobhanmoosavi/us-accidents",
            help="Format: username/dataset-name"
        )
        
        
        # File paths
        model_file = st.text_input("Model Path", "/Users/yashkailasdeshmane/Desktop/Final_usa_accidents_project/model/accident_severity_model.pkl")
        explainer_file = st.text_input("SHAP Explainer Path", "/Users/yashkailasdeshmane/Desktop/Final_usa_accidents_project/model/accident_severity_explainer.pkl")
        features_file = st.text_input("Features List Path", "/Users/yashkailasdeshmane/Desktop/Final_usa_accidents_project/model/feature_names.pkl")
        
        st.caption("Note: Ensure all files are correctly located in the paths specified.")
        
        # Refresh button
        if st.button("Reload Data and Models", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()
            
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This dashboard analyzes US traffic accidents and provides a severity prediction tool.")
        st.markdown("Data source: Kaggle US Accidents Dataset")
    
    # Load data and models
    data = load_data(kaggle_dataset)
    model = load_model(model_file)
    explainer = load_explainer(explainer_file)
    features_list = get_model_features(features_file)
    
    # Store weather options in session state if available
    if data is not None and 'Weather_Condition' in data.columns:
        st.session_state.weather_options = sorted(data['Weather_Condition'].unique().tolist())
        
    # Store feature columns in session state
    if data is not None:
        st.session_state.feature_columns = data.columns.tolist()
    
    # Check if data and models are loaded
    if data is None or model is None or explainer is None or features_list is None:
        st.error("Error: Could not load data or models. Please check that all files exist and are correctly specified.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Risk-Time Matrix", "Severity Prediction", "Data Exploration"])
    
    # Tab 1: Risk-Time Matrix
    with tab1:
        st.markdown('<h2 class="sub-header">Accident Location Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Filter Options")
            time_range = st.slider("Time Range (Hour)", 0, 23, (6, 22))
            
            day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_filter = st.multiselect("Day of Week", options=day_options, default=day_options)
            
            weather_options = st.session_state.get('weather_options', ['Clear', 'Cloudy', 'Rain'])
            weather_filter = st.multiselect("Weather Condition", options=weather_options, default=weather_options[:3] if len(weather_options) > 3 else weather_options)
            
            view_type = st.radio("View Type", options=["Points Map", "Density Heatmap"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display some statistics
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Key Statistics")
            
            # Calculate filtered statistics
            filtered_df = data.copy()
            if time_range:
                filtered_df = filtered_df[(filtered_df['Hour'] >= time_range[0]) & (filtered_df['Hour'] <= time_range[1])]
            if day_filter:
                filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(day_filter)]
            if weather_filter and 'Weather_Condition' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Weather_Condition'].isin(weather_filter)]
            
            total_accidents = len(filtered_df)
            avg_severity = filtered_df['Severity'].mean()
            peak_hour = filtered_df.groupby('Hour').size().idxmax()
            
            st.metric("Total Accidents", f"{total_accidents:,}")
            st.metric("Average Severity", f"{avg_severity:.2f}")
            st.metric("Peak Hour", f"{peak_hour}:00")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if view_type == "Points Map":
                accident_map = create_accident_map(data, time_range, day_filter, weather_filter)
                st.plotly_chart(accident_map, use_container_width=True)
            else:
                density_map = create_heatmap_layer(data, time_range, day_filter, weather_filter)
                st.plotly_chart(density_map, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional analysis
        st.markdown('<h3 class="sub-header">Time-based Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Hourly trend
            hourly_data = filtered_df.groupby('Hour').agg(
                Avg_Severity=('Severity', 'mean'),
                Count=('ID', 'count')
            ).reset_index()
            
            hourly_fig = px.line(hourly_data, x='Hour', y=['Avg_Severity', 'Count'], 
                                title='Hourly Accident Trend',
                                labels={'value': 'Value', 'variable': 'Metric'},
                                color_discrete_map={'Avg_Severity': '#1f77b4', 'Count': '#ff7f0e'})
            
            st.plotly_chart(hourly_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Day of week trend
            dow_data = filtered_df.groupby('Day_of_Week').agg(
                Avg_Severity=('Severity', 'mean'),
                Count=('ID', 'count')
            ).reset_index()
            
            # Ensure proper ordering of days
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_data['Day_of_Week'] = pd.Categorical(dow_data['Day_of_Week'], categories=dow_order, ordered=True)
            dow_data = dow_data.sort_values('Day_of_Week')
            
            dow_fig = px.bar(dow_data, x='Day_of_Week', y='Count', 
                             title='Day of Week Accident Trend',
                             color='Avg_Severity', color_continuous_scale='blues')
            
            st.plotly_chart(dow_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Severity Prediction
    with tab2:
        st.markdown('<h2 class="sub-header">Accident Severity Prediction Tool</h2>', unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
        <div class="highlight">
        This tool predicts the severity of a potential accident based on various factors. 
        Enter the details below to get a severity prediction and risk mitigation suggestions.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Show model performance
            st.subheader("Model Information")
            
            # Display model details
            st.markdown("""
            <b>Model:</b> XGBoost Classifier<br>
            <b>Features:</b> Time of day, location, weather conditions, and other factors<br>
            <b>Severity Levels:</b>
            <ul>
                <li><span class="severity-1">1 - Minor</span>: Slight delay, minimal impact on traffic</li>
                <li><span class="severity-2">2 - Moderate</span>: Noticeable delay, moderate impact</li>
                <li><span class="severity-3">3 - Significant</span>: Substantial delay, significant impact</li>
                <li><span class="severity-4">4 - Severe</span>: Major delay, severe impact on traffic</li>
            </ul>
            """, unsafe_allow_html=True)
            
            # Create prediction form
            input_data = create_prediction_form()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if input_data['submit']:
                # Add a loading indicator
                with st.spinner("Calculating prediction..."):
                    # Prepare the data for prediction
                    prediction_data = prepare_prediction_data(input_data, features_list)
                    
                    # Make prediction
                    try:
                        prediction = model.predict(prediction_data)[0]
                        # Add 1 to the prediction if the model was trained on 0-indexed labels
                        if prediction in [0, 1, 2, 3]:
                            prediction = prediction + 1
                            
                        probabilities = model.predict_proba(prediction_data)[0]
                        
                        # Display prediction result
                        st.subheader("Prediction Results")
                        
                        # Create a gauge chart for severity
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Predicted Severity"},
                            gauge={
                                'axis': {'range': [1, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [1, 2], 'color': "#BFDBFE"},  # Light blue
                                    {'range': [2, 3], 'color': "#93C5FD"},  # Medium blue
                                    {'range': [3, 4], 'color': "#60A5FA"}   # Dark blue
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prediction
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display severity description
                        severity_descriptions = {
                            1: "Minor impact on traffic flow with short delay",
                            2: "Moderate impact on traffic flow with noticeable delay",
                            3: "Significant impact on traffic flow with substantial delay",
                            4: "Severe impact on traffic flow with long delay"
                        }
                        
                        st.markdown(f"""
                        <div class="severity-{int(prediction)}">
                            <b>Prediction:</b> Severity Level {int(prediction)} - {severity_descriptions[int(prediction)]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display probability distribution
                        st.subheader("Severity Probability Distribution")
                        
                        # Adjust probabilities if the model was trained on 0-indexed labels
                        labels = [1, 2, 3, 4] if prediction in [1, 2, 3, 4] else [1, 2, 3, 4] 
                        
                        prob_df = pd.DataFrame({
                            'Severity': labels,
                            'Probability': probabilities * 100  # Convert to percentage
                        })
                        
                        prob_fig = px.bar(prob_df, x='Severity', y='Probability', 
                                        title='Severity Probability Distribution',
                                        color='Severity', 
                                        color_discrete_map={1: '#BFDBFE', 2: '#93C5FD', 3: '#60A5FA', 4: '#2563EB'},
                                        text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%'))
                        
                        prob_fig.update_layout(
                            xaxis_title="Severity Level",
                            yaxis_title="Probability (%)",
                            yaxis_range=[0, 100]
                        )
                        
                        st.plotly_chart(prob_fig, use_container_width=True)
                        
                        # Provide mitigation suggestions
                        st.subheader("Risk Mitigation Suggestions")
                        suggestions = provide_mitigation_suggestions(int(prediction), input_data)
                        
                        for i, suggestion in enumerate(suggestions):
                            st.markdown(f"• {suggestion}")
                        
                        # Display feature importance
                        importance_fig = plot_feature_importance(prediction_data, model, explainer)
                        st.pyplot(importance_fig)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("Please check that the model is correctly loaded and the input data is properly formatted.")
    # Tab 3: Data Exploration
    with tab3:
        st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        # Create columns for the analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Filter Dataset")
            
            # Create filters for data exploration
            severity_filter = st.multiselect("Severity", options=[1, 2, 3, 4], default=[1, 2, 3, 4])
            date_range = st.date_input("Date Range", 
                                       value=[data['Start_Time'].min().date(), data['Start_Time'].max().date()],
                                       min_value=data['Start_Time'].min().date(),
                                       max_value=data['Start_Time'].max().date())
            
            # Additional filters
            if 'State' in data.columns:
                state_options = sorted(data['State'].unique().tolist())
                state_filter = st.multiselect("State", options=state_options, default=state_options[:5] if len(state_options) > 5 else state_options)
            else:
                state_filter = None
                
            # Apply filters
            filtered_data = data.copy()
            filtered_data = filtered_data[filtered_data['Severity'].isin(severity_filter)]
            
            if len(date_range) == 2:
                filtered_data = filtered_data[(filtered_data['Start_Time'].dt.date >= date_range[0]) & 
                                             (filtered_data['Start_Time'].dt.date <= date_range[1])]
            
            if state_filter is not None:
                filtered_data = filtered_data[filtered_data['State'].isin(state_filter)]
                
            # Show sample of filtered data
            st.subheader("Sample Data")
            st.dataframe(filtered_data.head(10))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Data Statistics")
            
            # Display basic statistics
            data_stats = filtered_data.describe().T
            st.dataframe(data_stats)
            
            # Display correlation matrix if numerical columns exist
            numerical_cols = filtered_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numerical_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = filtered_data[numerical_cols].corr()
                
                # Create a heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', ax=ax)
                st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional visualizations
        st.markdown('<h3 class="sub-header">Advanced Visualizations</h3>', unsafe_allow_html=True)
        
        visualization_type = st.selectbox(
            "Select Visualization Type",
            ["Severity Distribution", "Monthly Trend", "Weather Impact", "Duration Analysis"]
        )
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if visualization_type == "Severity Distribution":
            # Severity distribution
            severity_counts = filtered_data['Severity'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            severity_counts = severity_counts.sort_values('Severity')
            
            fig = px.bar(severity_counts, x='Severity', y='Count',
                        title='Accident Severity Distribution',
                        color='Severity',
                        color_discrete_map={1: '#BFDBFE', 2: '#93C5FD', 3: '#60A5FA', 4: '#2563EB'})
            
            fig.update_layout(xaxis_title="Severity Level", yaxis_title="Number of Accidents")
            st.plotly_chart(fig, use_container_width=True)
            
        elif visualization_type == "Monthly Trend":
            # Monthly trend
            filtered_data['Year_Month'] = filtered_data['Start_Time'].dt.strftime('%Y-%m')
            monthly_data = filtered_data.groupby('Year_Month').agg(
                Count=('ID', 'count'),
                Avg_Severity=('Severity', 'mean')
            ).reset_index()
            
            fig = px.line(monthly_data, x='Year_Month', y=['Count', 'Avg_Severity'],
                         title='Monthly Accident Trend',
                         labels={'value': 'Value', 'variable': 'Metric'},
                         color_discrete_map={'Count': '#1E3A8A', 'Avg_Severity': '#2563EB'})
            
            fig.update_layout(xaxis_title="Month", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
            
        elif visualization_type == "Weather Impact":
            # Weather impact on accidents
            if 'Weather_Condition' in filtered_data.columns:
                top_weather = filtered_data['Weather_Condition'].value_counts().nlargest(10).index.tolist()
                weather_data = filtered_data[filtered_data['Weather_Condition'].isin(top_weather)]
                
                weather_severity = weather_data.groupby('Weather_Condition').agg(
                    Count=('ID', 'count'),
                    Avg_Severity=('Severity', 'mean')
                ).reset_index().sort_values('Count', ascending=False)
                
                fig = px.bar(weather_severity, x='Weather_Condition', y='Count',
                            title='Weather Impact on Accidents',
                            color='Avg_Severity',
                            color_continuous_scale='blues')
                
                fig.update_layout(xaxis_title="Weather Condition", yaxis_title="Number of Accidents")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Weather condition data not available in the dataset.")
                
        elif visualization_type == "Duration Analysis":
            # Duration analysis
            if 'Duration' in filtered_data.columns:
                # Create duration bins
                duration_bins = [0, 15, 30, 60, 120, 240, 480, float('inf')]
                duration_labels = ['0-15', '15-30', '30-60', '60-120', '120-240', '240-480', '480+']
                
                filtered_data['Duration_Bin'] = pd.cut(filtered_data['Duration'], bins=duration_bins, labels=duration_labels)
                
                duration_data = filtered_data.groupby('Duration_Bin').agg(
                    Count=('ID', 'count'),
                    Avg_Severity=('Severity', 'mean')
                ).reset_index()
                
                fig = px.bar(duration_data, x='Duration_Bin', y='Count',
                            title='Accident Duration Analysis',
                            color='Avg_Severity',
                            color_continuous_scale='blues')
                
                fig.update_layout(xaxis_title="Duration (minutes)", yaxis_title="Number of Accidents")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Duration data not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()                    

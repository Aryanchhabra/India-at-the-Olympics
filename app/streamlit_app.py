"""
India at the Olympics: Medal Prediction & Insights
Streamlit Dashboard for Interactive Data Exploration and Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="India Olympics Dashboard",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load all processed data files"""
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        india_df = pd.read_csv(os.path.join(script_dir, 'india_olympics_data.csv'))
        yearly_data = pd.read_csv(os.path.join(script_dir, 'yearly_aggregated_data.csv'))
        top_athletes = pd.read_csv(os.path.join(script_dir, 'top_athletes.csv'))
        
        # Load sports_medals and ensure it's a Series
        sports_medals_df = pd.read_csv(os.path.join(script_dir, 'sports_medals.csv'), index_col=0)
        if isinstance(sports_medals_df, pd.DataFrame):
            # If it's a DataFrame, get the first column as a Series
            sports_medals = sports_medals_df.iloc[:, 0]
        else:
            sports_medals = sports_medals_df
        
        return india_df, yearly_data, top_athletes, sports_medals
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Please run the Jupyter notebook first to generate the data files.")
        st.stop()

@st.cache_resource
def load_models():
    """Load trained ML models and metadata"""
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        lr_model = joblib.load(os.path.join(script_dir, 'lr_model.pkl'))
        rf_model = joblib.load(os.path.join(script_dir, 'rf_model.pkl'))
        scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))
        metadata = joblib.load(os.path.join(script_dir, 'model_metadata.pkl'))
        return lr_model, rf_model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please run the Jupyter notebook first to train and save the models.")
        st.stop()

# Load all data
india_df, yearly_data, top_athletes, sports_medals = load_data()
lr_model, rf_model, scaler, metadata = load_models()

# Main header
st.markdown('<p class="main-header">🏅 India at the Olympics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Medal Prediction & Insights Dashboard | Summer Olympics (1948-Present)</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

# Sidebar filters
st.sidebar.subheader("📊 Data Filters")

# Year range filter
min_year = int(yearly_data['Year'].min())
max_year = int(yearly_data['Year'].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=4
)

# Sports filter
all_sports = sorted(india_df['Sport'].unique())
selected_sports = st.sidebar.multiselect(
    "Select Sports",
    options=all_sports,
    default=all_sports[:5] if len(all_sports) > 5 else all_sports
)

# Model selection for predictions
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose Prediction Model",
    options=["Random Forest", "Linear Regression"],
    index=0
)

# Filter data based on selections
filtered_yearly = yearly_data[
    (yearly_data['Year'] >= year_range[0]) & 
    (yearly_data['Year'] <= year_range[1])
]

filtered_india = india_df[
    (india_df['Year'] >= year_range[0]) & 
    (india_df['Year'] <= year_range[1])
]

if selected_sports:
    filtered_india = filtered_india[filtered_india['Sport'].isin(selected_sports)]

# Key Statistics Cards
st.markdown("### 📈 Key Statistics")
st.info("ℹ️ **Note:** Showing 23 actual unique medals (1948-2016). The dataset has 140 athlete records because team sports count each player separately.")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    # Calculate ACTUAL unique medals (not athlete records!)
    medal_records = india_df[india_df['Medal'] != 'No Medal']
    unique_medals_count = medal_records.groupby(['Year', 'Sport', 'Event', 'Medal']).size()
    total_medals = len(unique_medals_count)
    st.metric("Total Medals Won", total_medals)

with col2:
    best_year = yearly_data.loc[yearly_data['Total_Medals'].idxmax(), 'Year']
    best_count = yearly_data['Total_Medals'].max()
    st.metric("Best Performance", f"{int(best_year)}", f"{int(best_count)} medals")

with col3:
    # Fix: Extract the sport name properly from the Series
    top_sport_name = str(sports_medals.idxmax())
    top_sport_medals = int(sports_medals.max())
    st.metric("Top Sport", top_sport_name, f"{top_sport_medals} medals")

with col4:
    total_athletes = india_df['Name'].nunique()
    st.metric("Total Athletes", total_athletes)

with col5:
    recent_year = int(yearly_data['Year'].max())
    recent_medals = int(yearly_data[yearly_data['Year'] == recent_year]['Total_Medals'].values[0])
    st.metric(f"{recent_year} Performance", f"{recent_medals} medals")

st.markdown("---")

# Create tabs for different sections
tabs = st.tabs(["📊 Historical Performance", "🔍 EDA Insights", "🤖 Medal Prediction", "📋 Data Explorer"])

# Tab 1: Historical Performance
with tabs[0]:
    st.header("Historical Performance Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Medal Trend Over the Years")
        fig = px.line(filtered_yearly, x='Year', y='Total_Medals',
                     title='India\'s Medal Count Evolution',
                     markers=True,
                     labels={'Total_Medals': 'Total Medals', 'Year': 'Olympic Year'})
        fig.update_traces(line_color='#FF6B35', marker=dict(size=10), 
                         line=dict(width=3))
        fig.update_layout(hovermode='x unified', template='plotly_white', height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Medal Distribution")
        # Calculate ACTUAL unique medals by type
        medal_records = india_df[india_df['Medal'] != 'No Medal']
        unique_medals = medal_records.groupby(['Year', 'Sport', 'Event', 'Medal']).size().reset_index(name='Count')
        medal_counts = unique_medals['Medal'].value_counts()
        fig = px.pie(values=medal_counts.values, names=medal_counts.index,
                    title='Actual Medal Type Breakdown (23 medals)',
                    color=medal_counts.index,
                    color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
        fig.update_layout(template='plotly_white', height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Participation Growth")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_yearly['Year'], 
            y=filtered_yearly['Male_Athletes'],
            mode='lines+markers',
            name='Male',
            line=dict(color='#4A90E2', width=3),
            stackgroup='one'
        ))
        fig.add_trace(go.Scatter(
            x=filtered_yearly['Year'],
            y=filtered_yearly['Female_Athletes'],
            mode='lines+markers',
            name='Female',
            line=dict(color='#E94B3C', width=3),
            stackgroup='one'
        ))
        fig.update_layout(
            title='Male vs Female Athletes Over Time',
            xaxis_title='Olympic Year',
            yaxis_title='Number of Athletes',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sports Diversity Evolution")
        fig = px.area(filtered_yearly, x='Year', y='Unique_Sports',
                     title='Number of Sports India Participated In',
                     labels={'Unique_Sports': 'Unique Sports', 'Year': 'Olympic Year'})
        fig.update_traces(line_color='#2ECC71', fillcolor='rgba(46, 204, 113, 0.3)')
        fig.update_layout(hovermode='x unified', template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: EDA Insights
with tabs[1]:
    st.header("Exploratory Data Analysis Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Sports by Medal Count")
        top_10_sports = sports_medals.sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_10_sports.values,
            y=top_10_sports.index,
            orientation='h',
            title='Most Successful Sports for India',
            labels={'x': 'Total Medals', 'y': 'Sport'},
            color=top_10_sports.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            showlegend=False,
            template='plotly_white',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Indian Athletes")
        top_10_athletes = top_athletes.head(10)
        fig = px.bar(
            top_10_athletes,
            x='Total_Medals',
            y='Name',
            orientation='h',
            title='Athletes with Most Medals',
            labels={'Total_Medals': 'Total Medals', 'Name': 'Athlete'},
            color='Total_Medals',
            color_continuous_scale='Reds',
            hover_data=['Sport', 'Olympic_Years']
        )
        fig.update_layout(
            showlegend=False,
            template='plotly_white',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decade-wise Performance")
        decade_data = filtered_yearly.copy()
        decade_data['Decade'] = (decade_data['Year'] // 10) * 10
        decade_performance = decade_data.groupby('Decade')['Total_Medals'].sum().reset_index()
        
        fig = px.bar(
            decade_performance,
            x='Decade',
            y='Total_Medals',
            title='Total Medals by Decade',
            labels={'Total_Medals': 'Total Medals', 'Decade': 'Decade'},
            text='Total_Medals'
        )
        fig.update_traces(marker_color='#3498DB', textposition='outside')
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Gender Diversity Index")
        fig = px.line(
            filtered_yearly,
            x='Year',
            y='Gender_Ratio',
            title='Female Athletes Ratio Over Time',
            markers=True,
            labels={'Gender_Ratio': 'Female Athletes / Total Athletes', 'Year': 'Olympic Year'}
        )
        fig.update_traces(line_color='#9B59B6', marker=dict(size=10))
        fig.update_layout(hovermode='x unified', template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Top Athletes Table")
    st.dataframe(
        top_athletes.head(20),
        use_container_width=True,
        hide_index=True
    )

# Tab 3: Medal Prediction
with tabs[2]:
    st.header("🤖 Medal Prediction System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Features")
        st.markdown("Adjust the parameters below to predict India's medal count:")
        
        # Input form
        with st.form("prediction_form"):
            pred_year = st.number_input(
                "Olympic Year",
                min_value=2024,
                max_value=2050,
                value=2028,
                step=4,
                help="Enter the Olympic year (must be a multiple of 4)"
            )
            
            pred_athletes = st.slider(
                "Total Athletes",
                min_value=50,
                max_value=200,
                value=120,
                help="Estimated number of Indian athletes"
            )
            
            pred_sports = st.slider(
                "Number of Sports",
                min_value=10,
                max_value=25,
                value=18,
                help="Number of different sports India will participate in"
            )
            
            pred_events = st.slider(
                "Number of Events",
                min_value=20,
                max_value=100,
                value=65,
                help="Number of different events"
            )
            
            pred_female = st.slider(
                "Female Athletes",
                min_value=20,
                max_value=100,
                value=60,
                help="Number of female athletes"
            )
            
            pred_prev_medals = st.number_input(
                "Previous Olympics Medals",
                min_value=0,
                max_value=50,
                value=7,
                help="Medals won in the previous Olympics"
            )
            
            pred_prev_2_medals = st.number_input(
                "2 Olympics Ago Medals",
                min_value=0,
                max_value=50,
                value=6,
                help="Medals won 2 Olympics ago"
            )
            
            submit_button = st.form_submit_button("🎯 Predict Medal Count", use_container_width=True)
        
    with col2:
        st.subheader("Prediction Results")
        
        if submit_button:
            # Calculate derived features
            gender_ratio = pred_female / pred_athletes
            years_since_1948 = pred_year - 1948
            olympic_count = (pred_year - 1948) // 4 + 1
            
            # Prepare feature vector
            features = np.array([[
                pred_athletes,
                pred_sports,
                pred_events,
                pred_female,
                gender_ratio,
                years_since_1948,
                olympic_count,
                pred_prev_medals,
                pred_prev_2_medals
            ]])
            
            # Make prediction based on selected model
            if selected_model == "Random Forest":
                prediction = rf_model.predict(features)[0]
                model_info = f"Random Forest (Test MAE: {metadata['rf_test_mae']:.2f}, R²: {metadata['rf_test_r2']:.4f})"
            else:
                features_scaled = scaler.transform(features)
                prediction = lr_model.predict(features_scaled)[0]
                model_info = f"Linear Regression (Test MAE: {metadata['lr_test_mae']:.2f}, R²: {metadata['lr_test_r2']:.4f})"
            
            # Ensure non-negative prediction
            prediction = max(0, round(prediction))
            
            # Display prediction
            st.success(f"### Predicted Medal Count for {pred_year}: **{prediction} medals** 🏅")
            st.info(f"**Model Used:** {model_info}")
            
            # Confidence interval (approximate)
            if selected_model == "Random Forest":
                std_error = metadata['rf_test_rmse']
            else:
                std_error = metadata['lr_test_rmse']
            
            lower_bound = max(0, int(prediction - std_error))
            upper_bound = int(prediction + std_error)
            
            st.markdown(f"**Confidence Range:** {lower_bound} - {upper_bound} medals")
            
            # Comparison with historical data
            st.markdown("---")
            st.markdown("#### 📊 Historical Context")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Historical Average", f"{yearly_data['Total_Medals'].mean():.1f}")
            with col_b:
                st.metric("Historical Best", f"{yearly_data['Total_Medals'].max():.0f}")
            with col_c:
                recent_avg = yearly_data.tail(3)['Total_Medals'].mean()
                st.metric("Recent Average (3 Olympics)", f"{recent_avg:.1f}")
        
        else:
            st.info("👈 Fill in the parameters and click 'Predict Medal Count' to see the prediction")
        
        # Show feature importance
        st.markdown("---")
        st.markdown("#### 🔍 Feature Importance")
        
        feature_imp = pd.DataFrame({
            'Feature': metadata['feature_columns'],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Random Forest Feature Importance',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
            color='Importance',
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance comparison
    st.markdown("---")
    st.subheader("📈 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²'],
            'Linear Regression': [
                metadata['lr_test_mae'],
                metadata['lr_test_rmse'],
                metadata['lr_test_r2']
            ],
            'Random Forest': [
                metadata['rf_test_mae'],
                metadata['rf_test_rmse'],
                metadata['rf_test_r2']
            ]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Show saved visualizations if available
        if os.path.exists('predicted_vs_actual.png'):
            try:
                img = Image.open('predicted_vs_actual.png')
                st.image(img, caption='Predicted vs Actual Performance', use_container_width=True)
            except:
                pass

# Tab 4: Data Explorer
with tabs[3]:
    st.header("📋 Data Explorer")
    
    st.markdown("### Filtered Dataset Overview")
    st.markdown(f"**Records:** {len(filtered_india):,} | **Year Range:** {year_range[0]} - {year_range[1]}")
    
    if selected_sports:
        st.markdown(f"**Selected Sports:** {', '.join(selected_sports)}")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_medals_only = st.checkbox("Show only medal winners", value=False)
    with col2:
        records_to_show = st.selectbox("Records to display", [50, 100, 200, 500, "All"], index=0)
    with col3:
        download_format = st.selectbox("Download format", ["CSV", "Excel"])
    
    # Filter for medals only if selected
    display_df = filtered_india.copy()
    if show_medals_only:
        display_df = display_df[display_df['Medal'] != 'No Medal']
    
    # Determine how many records to show
    if records_to_show == "All":
        records_to_show = len(display_df)
    else:
        records_to_show = int(records_to_show)
    
    # Display dataframe
    st.dataframe(
        display_df.head(records_to_show),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if download_format == "CSV":
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f'india_olympics_{year_range[0]}_{year_range[1]}.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            # For Excel download (requires openpyxl)
            st.info("Excel download requires openpyxl. Download CSV instead.")
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Medal Statistics**")
        medals_won = display_df[display_df['Medal'] != 'No Medal']
        st.write(f"Total medals: {len(medals_won)}")
        st.write(f"Unique athletes with medals: {medals_won['Name'].nunique()}")
        st.write(f"Medal rate: {(len(medals_won)/len(display_df)*100):.2f}%")
    
    with col2:
        st.markdown("**Participation Statistics**")
        st.write(f"Total athletes: {display_df['Name'].nunique()}")
        st.write(f"Sports covered: {display_df['Sport'].nunique()}")
        st.write(f"Events participated: {display_df['Event'].nunique()}")
    
    with col3:
        st.markdown("**Gender Distribution**")
        gender_dist = display_df['Sex'].value_counts()
        st.write(f"Male: {gender_dist.get('M', 0)}")
        st.write(f"Female: {gender_dist.get('F', 0)}")
        if len(display_df) > 0:
            st.write(f"Female %: {(gender_dist.get('F', 0)/len(display_df)*100):.1f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>India at the Olympics: Medal Prediction & Insights</strong></p>
        <p>Data Source: Kaggle - 120 Years of Olympic History | Analysis Period: 1948-Present (Summer Olympics Only)</p>
        <p>Built with Streamlit, Plotly, and scikit-learn | Machine Learning Models: Linear Regression & Random Forest</p>
    </div>
""", unsafe_allow_html=True)




# 🏅 India at the Olympics: Medal Prediction & Insights

An interactive data science dashboard analyzing 70+ years of India's Summer Olympics performance (1948-2016) with machine learning-powered medal predictions.

[![Live Dashboard](https://img.shields.io/badge/🌐_Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://india-at-the-olympics.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **[🎯 View Live Application](https://india-at-the-olympics.streamlit.app/)** — Explore interactive visualizations and ML predictions

---

## 🎯 Project Overview

This project provides comprehensive analysis of India's **23 unique medals** across 18 Summer Olympics (1948-2016) and leverages machine learning to predict future performance trends.

### Key Features
- **Interactive Data Visualizations**: Explore India's Olympic journey through dynamic charts and graphs
- **ML-Powered Predictions**: Dual-model approach using Linear Regression and Random Forest algorithms
- **Sport-Specific Analysis**: Deep dive into performance across Hockey, Wrestling, Shooting, and more
- **Gender Participation Trends**: Track the evolution of female athlete representation over 70 years
- **Comprehensive Statistics**: Year-by-year medal breakdowns with athlete-level insights

---

## 🚀 Running Locally

To run this project on your local machine:

```bash
# Clone the repository
git clone https://github.com/Aryanchhabra/India-at-the-Olympics.git
cd India-at-the-Olympics

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook to generate models and processed data
jupyter notebook notebooks/olympic_analysis.ipynb
# Execute all cells in the notebook

# Launch the Streamlit dashboard
cd app
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Key Insights & Statistics

### Medal Performance (1948-2016)
- **Total Medals**: 23 unique medals (6 Gold, 5 Silver, 12 Bronze)
- **Peak Performance**: 2012 London Olympics — 6 medals
- **Dominant Sport**: Hockey with 8 medals (1948-1980 golden era)
- **Total Athletes**: 779 unique athletes represented India across all games

### Top Performing Sports
| Rank | Sport | Medals |
|------|-------|--------|
| 1 | Hockey | 8 |
| 2 | Wrestling | 5 |
| 3 | Shooting | 4 |
| 4 | Badminton | 2 |
| 5 | Boxing | 2 |

### Gender Participation Evolution
- **1948**: <5% female athletes
- **2016**: 30%+ female athletes
- Significant growth in women's representation across all sports

> **Note**: The dataset contains 140 athlete records as team sports count each player individually, resulting in 23 actual unique medal events.

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Development** | Jupyter Notebook |

---

## 📁 Project Structure

```
India-at-the-Olympics/
├── data/                          # Raw Olympic datasets
│   ├── athlete_events.csv         # Historical athlete data
│   ├── noc_regions.csv            # NOC region mappings
│   └── README.md                  # Data documentation
├── notebooks/                     # Analysis notebooks
│   └── olympic_analysis.ipynb     # Complete EDA and model training
├── app/                           # Production application
│   ├── streamlit_app.py           # Main dashboard application
│   ├── *.pkl                      # Trained ML models
│   └── *.csv                      # Processed datasets
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 📈 Machine Learning Models

### Model Architecture
Two complementary models provide predictions with varying approaches:

| Model | Type | Role | Performance |
|-------|------|------|-------------|
| **Linear Regression** | Baseline | Simple linear relationships | MAE: ~3.5 medals |
| **Random Forest** | Primary | Captures complex patterns | MAE: <3 medals |

### Features Engineering
- **Participation Metrics**: Total athletes, unique sports count
- **Gender Diversity**: Female athlete ratio, gender distribution
- **Historical Performance**: Lag features from previous Olympics
- **Temporal Features**: Years since independence, Olympic edition number

### Model Evaluation
- **Mean Absolute Error (MAE)**: Measures average prediction deviation
- **R² Score**: Explains variance in medal count predictions
- **Cross-Validation**: Leave-last-Olympics-out validation strategy

---

## 🔍 Dataset Information

**Source**: [120 Years of Olympic History: Athletes and Results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) (Kaggle)

**Scope**:
- **Total Records**: 271,116 athlete records (1896-2016)
- **Filtered For**: India · Summer Olympics · Post-Independence (1948+)
- **Coverage Period**: 18 Olympic Games (1948-2016)

**Limitations**: Dataset does not include 2020 Tokyo Olympics (6 medals) and 2024 Paris Olympics (6 medals)

---

## 🎨 Dashboard Capabilities

The interactive Streamlit dashboard offers four comprehensive sections:

### 1. Historical Performance
- **Medal Trends**: Visualize India's medal count trajectory from 1948-2016
- **Gender Analysis**: Track male vs. female athlete participation over time
- **Sport Diversity**: Monitor the expansion of India's Olympic sports portfolio

### 2. EDA Insights
- **Top Performers**: Identify highest-performing sports and individual athletes
- **Decade Analysis**: Compare performance across different Olympic eras
- **Participation Metrics**: Analyze athlete counts, sport diversity, and success rates

### 3. Medal Prediction
- **Interactive Inputs**: Customize parameters (year, athletes, sports, gender ratio)
- **Dual Predictions**: Compare Linear Regression and Random Forest outputs
- **Model Insights**: View feature importance and prediction confidence intervals
- **Visual Comparison**: Predicted vs. actual performance charts

### 4. Data Explorer
- **Advanced Filtering**: Filter dataset by year, sport, medal type, gender
- **Data Export**: Download filtered datasets as CSV
- **Statistical Summary**: View descriptive statistics for any data subset

---

## 💡 Future Enhancements

Potential areas for project expansion:
- **Updated Data**: Integration of 2020 Tokyo and 2024 Paris Olympics results
- **Economic Factors**: Incorporate GDP and sports funding data for enhanced predictions
- **Sport-Specific Models**: Individual prediction models for each sport category
- **Explainability**: Implementation of SHAP values for model interpretability
- **Real-Time Updates**: Automated data pipeline for live Olympic coverage

---

## 👨‍💻 Author

**Aryan Chhabra**  
Data Science & Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-@Aryanchhabra-181717?style=flat&logo=github)](https://github.com/Aryanchhabra)
[![Portfolio](https://img.shields.io/badge/Project-India_at_Olympics-FF4B4B?style=flat&logo=github)](https://github.com/Aryanchhabra/India-at-the-Olympics)

---

## 📄 License

This project is licensed under the MIT License - free to use for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- **Dataset Source**: [Kaggle - rgriffin](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)
- **Data Attribution**: Sports-Reference.com, International Olympic Committee
- **Inspiration**: India's remarkable Olympic journey and growing sports culture 🇮🇳

---

<div align="center">

**⭐ If you found this project helpful, consider starring the repository!**

*Built with Python · Streamlit · Scikit-learn*

</div>

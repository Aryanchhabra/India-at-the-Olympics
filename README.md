# 🏅 India at the Olympics: Medal Prediction & Insights

An interactive data science dashboard analyzing India's Summer Olympics performance (1948-2016) with machine learning predictions.

[![Live Dashboard](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)](https://india-at-the-olympics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **[🚀 View Live Dashboard](https://india-at-the-olympics.streamlit.app)** | **[📊 Explore the Data](data/)** | **[📓 View Analysis](notebooks/)**

---

## 🎯 Project Overview

This project analyzes India's **23 unique medals** from 18 Summer Olympics (1948-2016) and uses machine learning to predict future performance.

**Key Features:**
- 📊 Interactive visualizations of India's Olympic journey
- 🤖 ML-powered medal predictions (Linear Regression & Random Forest)
- 🏑 Sport-wise performance analysis (Hockey, Wrestling, Shooting, etc.)
- 👥 Gender participation trends over 70 years
- 📈 Year-by-year medal breakdowns

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Aryanchhabra/India-at-the-Olympics.git
cd India-at-the-Olympics

# Install dependencies
pip install -r requirements.txt

# Run the analysis notebook (generates models and processed data)
jupyter notebook
# Open notebooks/olympic_analysis.ipynb and run all cells

# Launch the dashboard
cd app
streamlit run streamlit_app.py
```

### View Online
**[🌐 Live Dashboard →](https://india-at-the-olympics.streamlit.app)**

---

## 📊 Key Insights

### Medal Count (1948-2016)
- **23 actual unique medals** (6 Gold, 5 Silver, 12 Bronze)
- **⚠️ Note:** Dataset shows 140 athlete records because team sports count each player separately

### Top Sports
1. 🏑 **Hockey** - 8 medals (dominated 1948-1980)
2. 🤼 **Wrestling** - 5 medals
3. 🎯 **Shooting** - 4 medals
4. 🏸 **Badminton** - 2 medals
5. 🥊 **Boxing** - 2 medals

### Key Statistics
- **779 unique athletes** represented India
- **23 different sports** participated in
- **Best performance:** 2012 London (6 medals)
- **Gender diversity:** Increased from <5% to 30%+ female athletes

---

## 🛠️ Tech Stack

- **Python 3.8+** - Core language
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - ML models (Linear Regression, Random Forest)
- **Plotly** - Interactive visualizations
- **Streamlit** - Web dashboard
- **Jupyter** - Data analysis

---

## 📁 Project Structure

```
Olympic/
├── data/                    # Olympics dataset (1948-2016)
├── notebooks/              # Jupyter analysis notebook
├── app/                    # Streamlit dashboard + models
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 📈 Machine Learning

**Models:**
- Linear Regression (baseline)
- Random Forest Regressor (primary)

**Target:** Total medals per Olympic year

**Features:** Athletes sent, sports participated, gender diversity, previous medals (lag features)

**Performance:** Random Forest achieves MAE < 3 medals on test set

---

## 🔍 Dataset

**Source:** [Kaggle - 120 Years of Olympic History](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)

**Coverage:**
- 271,116 athlete records (1896-2016)
- Filtered for India, Summer Olympics, 1948+
- ⚠️ Missing: 2020 Tokyo (6 medals) & 2024 Paris (6 medals)

---

## 🎨 Dashboard Features

### 📊 Historical Performance
- Medal trends from 1948-2016
- Gender participation evolution
- Sports diversity over time

### 🔍 EDA Insights
- Top performing sports and athletes
- Decade-wise analysis
- Gender diversity trends

### 🤖 Medal Prediction
- Interactive form with custom parameters
- Predictions from both ML models
- Feature importance visualization
- Confidence intervals

### 📋 Data Explorer
- Filterable dataset viewer
- CSV download functionality
- Summary statistics

---

## 🚀 Deployment

### Streamlit Cloud (Current)

1. Push to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub repository
4. Deploy from `app/streamlit_app.py`

**Live at:** [india-at-the-olympics.streamlit.app](https://india-at-the-olympics.streamlit.app)

### Local Development

```bash
cd app
streamlit run streamlit_app.py
```

---

## 🤝 Contributing

Contributions welcome! Ideas for enhancement:
- Add 2020 Tokyo & 2024 Paris Olympics data
- Incorporate GDP/funding data for better predictions
- Build sport-specific prediction models
- Add SHAP for model explainability

---

## 📄 License

MIT License - feel free to use this project for learning and portfolio purposes.

---

## 👨‍💻 Author

**Aryan Chhabra**
- GitHub: [@Aryanchhabra](https://github.com/Aryanchhabra)
- Project: [India at the Olympics](https://github.com/Aryanchhabra/India-at-the-Olympics)

---

## 🙏 Acknowledgments

- **Dataset:** [rgriffin/Kaggle](https://www.kaggle.com/heesoo37)
- **Data Source:** sports-reference.com, International Olympic Committee
- **Inspiration:** India's growing Olympic success story 🇮🇳

---

**⭐ Star this repo if you found it helpful!**

*Last Updated: October 2025*

# 🦠 Epidemic Outbreak Prediction & Risk Analysis

An AI-powered system to **predict infectious disease spread**, identify **high-risk regions**, and visualize outbreaks through an **interactive dashboard**.

---

## 🚀 Project Overview

Predicting disease outbreaks is critical for public health preparedness.
This project combines **time-series modeling + mobility data + visualization** to deliver:

* 📈 Accurate outbreak prediction
* 🌍 Global hotspot detection
* 🚨 Risk classification (Low / Medium / High)
* 🧭 Interactive epidemic dashboard

---

## 🧠 Key Features

* ✅ Time-series forecasting of daily cases
* ✅ Integration of **Google Mobility Data**
* ✅ Advanced feature engineering (growth rate, lag features, rolling averages)
* ✅ Risk classification system
* ✅ Interactive dashboard with global and country-level insights
* ✅ 7-day outbreak forecasting

---

## 📊 Dataset

### 1. Epidemiological Data

* Country-wise time-series case data
* Columns: `Country`, `Date`, `Cases`

### 2. Mobility Data

* Google Mobility Reports
* Includes:

  * Retail & recreation
  * Workplaces
  * Transit stations
  * Residential mobility

---

## 📂 Dataset Access

Due to large file sizes, datasets are not included in this repository.

👉 **Download Dataset Here:**
[🔗 Download Dataset](https://drive.google.com/drive/folders/173HyVxAQKgn101-Wpc8XE0gsmxY1jFMs?usp=sharing)

---

### 📁 Folder Structure (Inside Dataset)

The dataset is already structured to match the project:

```
data/
├── raw/
│   ├── raw_data.csv
│   ├── mobility.csv
│
├── processed/
│   ├── processed_data.csv
│   ├── merged_data.csv
│   ├── featured_data.csv
```

---

### ⚙️ Setup Instructions

1. Download the dataset from the link above
2. Extract the files
3. Copy the entire `data/` folder
4. Replace the existing `data/` folder in this project

---

### ⚠️ Note

* This repository may contain **sample CSV files** for demonstration
* For full functionality and accurate results, replace them with the dataset provided

---

### 🔁 Optional: Re-run Pipeline

If you want to regenerate processed data from raw files:

```bash
python src/preprocess.py
python src/merge_external.py
python src/features.py
```

---

### ✅ Quick Start (Recommended)

To directly run the dashboard:

```bash
streamlit run app/dashboard.py
```

---

## ⚙️ Tech Stack

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **Streamlit**
* **Plotly**
* **Joblib**

---

## 🏗️ Project Structure

```
epidemic-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── preprocess.py
│   ├── merge_external.py
│   ├── features.py
│   ├── train_model.py
│
├── app/
│   └── dashboard.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow Pipeline

```
Raw Data 
   ↓
Preprocessing 
   ↓
Mobility Integration 
   ↓
Feature Engineering 
   ↓
Model Training 
   ↓
Prediction & Visualization
```

---

## 📈 Model Details

* Model: **Random Forest Regressor**
* Target: Daily New Cases

### Key Features Used:

* Growth Rate
* 7-Day Average Cases
* Lag Features (1-day, 7-day)
* Mobility Indicators

---

## 📊 Performance Metrics

* **MAE:** ~1110
* **RMSE:** ~13262
* **MAPE:** ~24.7%

> The model captures outbreak trends effectively while maintaining stability across countries.

---

## 🔥 Key Insights

* Growth rate is the **strongest indicator of outbreak spread**
* Recent trends (7-day averages) significantly influence predictions
* Mobility patterns provide valuable behavioral context

---

## 🖥️ Dashboard Features

### 🌍 Global Insights

* Animated outbreak spread map (timeline visualization)
* Top 10 high-risk countries
* Hotspot analysis (growth vs cases)

### 📊 Country Analysis

* Historical case trends
* Next-day prediction
* 7-day outbreak forecast
* Risk level classification

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/epidemic-prediction.git
cd epidemic-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline (optional)

```bash
python src/preprocess.py
python src/merge_external.py
python src/features.py
python src/train_model.py
```

### 4. Launch dashboard

```bash
streamlit run app/dashboard.py
```

---

## 🎯 Use Cases

* Public health monitoring
* Early outbreak detection
* Resource allocation planning
* Policy decision support

---

## 🏆 What Makes This Project Stand Out

* 🔗 Combines **epidemiological + mobility data**
* 🔮 Predictive modeling with real-world signals
* 🌍 Interactive and intuitive visualizations
* 📊 End-to-end machine learning pipeline

---

## 🔮 Future Improvements

* Deep learning models (LSTM, Transformer)
* Weather & demographic integration
* Real-time API deployment
* Region-level predictions (state/city level)

---

## 👩‍💻 Authors

**Richa Mishra**
**Sanskriti Sontakke**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

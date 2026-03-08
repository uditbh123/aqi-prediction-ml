# Air Quality Index (AQI) Prediction
### Machine Learning Course — Final Project

**Course:** Introduction to Machine Learning  
**Project Type:** Regression  
**Team Members:** Udit, Milan, Dipen, Lotus  

---

## Project Overview

Air pollution kills an estimated **7 million people per year** according to the World Health Organisation — making it the world's largest environmental health risk.

In this project we built a machine learning model that predicts the **Air Quality Index (AQI)** from pollutant concentration measurements. This kind of model can power early warning systems, help city planners identify pollution hotspots, and enable smart home devices to respond automatically to dangerous air quality.

---

## Project Files

| File | Description |
|---|---|
| `AQI_Prediction_Project.ipynb` | Main Jupyter Notebook — full ML pipeline |
| `aqi_dataset.csv` | Generated dataset — 1,500 air quality readings |
| `README.md` | This summary file |

---

## Dataset

- **Source:** Synthetically generated to mirror real monitoring station data
- **Samples:** 1,500 air quality readings
- **Features:** 9 input features + 1 target variable

| Feature | Description | Unit |
|---|---|---|
| `PM2.5` | Fine particulate matter — from vehicle exhaust & wildfires | µg/m³ |
| `PM10` | Coarse particulate matter — from road dust & construction | µg/m³ |
| `NO2` | Nitrogen Dioxide — from burning fuel in cars & power plants | ppb |
| `SO2` | Sulfur Dioxide — from coal-burning factories | ppb |
| `CO` | Carbon Monoxide — from incomplete combustion | ppm |
| `O3` | Ground-level Ozone — forms when sunlight reacts with NO2 | ppb |
| `Temperature` | Ambient air temperature | °C |
| `Humidity` | Relative humidity | % |
| `Wind_Speed` | Wind speed — the only feature that **reduces** AQI | km/h |
| `AQI` | **Target variable** — Air Quality Index | 0–500 |

---

## ML Pipeline

```
1. DEFINE PROBLEM    →  Predict AQI (regression)
2. COLLECT DATA      →  Generate 1,500 realistic readings → save as CSV → load
3. EXPLORE (EDA)     →  Distributions, correlations, AQI categories
4. PREPROCESS        →  Handle missing values, keep outliers, scale features
5. SPLIT DATA        →  80% train / 20% test
6. TRAIN MODELS      →  Linear Regression, KNN, SVM
7. EVALUATE          →  MAE, RMSE, R² for all models
8. ANALYSE           →  Feature coefficients, residuals, cross-validation
9. CONCLUDE          →  Best model identified, real-world implications
```

---

## Preprocessing Decisions

| Step | Method | Reason |
|---|---|---|
| Missing values | Median imputation | Robust against outliers — more honest than mean for skewed data |
| Outliers | **Kept intentionally** | Pollution spikes are real events (wildfires, accidents) — not errors |
| Feature scaling | StandardScaler | Brings all features to same scale — essential for KNN and SVM |
| Split | 80/20 train/test | Standard for medium datasets — 1,200 training, 300 test samples |

> **Key insight:** Removing outliers with IQR method dropped R² to 0.736. Keeping them improved it to **0.853** — a 16.2% improvement. Domain knowledge matters as much as technical skill.

---

## Models & Results

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| **Linear Regression**  | **0.8526** | **13.46** | best |
| SVM (RBF Kernel) | 0.8300 | 14.70 | — |
| K-Nearest Neighbors | 0.8019 | 15.61 | — |

### Why Linear Regression Won
AQI is defined as a weighted linear sum of pollutants — a fundamentally linear equation. Linear Regression is specifically designed to find this exact pattern. SVM and KNN are more powerful for complex non-linear relationships, but here the underlying pattern is linear so extra complexity brings no benefit.

### What the Metrics Mean
- **R² = 0.8526** → model explains 85.3% of why AQI goes up and down
- **MAE = 13.46** → on average, predictions are off by ~13 AQI points (¼ of one health category)
- **Remaining 14.7%** → genuine random noise added to simulate real sensor uncertainty — mathematically impossible to predict

---

##  Key Findings

1. **PM10 is the strongest predictor** of AQI according to Linear Regression coefficients
2. **Wind Speed negatively correlates** with AQI — stronger wind disperses pollution, reducing AQI
3. **High PM2.5 + CO together** are strong indicators of hazardous air quality conditions
4. **Keeping outliers improved R²** from 0.736 → 0.853 — pollution spikes must be included
5. **Linear Regression outperformed** more complex models because AQI is a linear relationship

---

##  Real-World Applications

-  Early warning systems for vulnerable populations (elderly, children, asthma patients)
-  City planning and industrial zone regulation
-  Smart home air purifier automation
-  Public health alert mobile applications
-  Environmental policy decision support

---

##  How to Run

1. Make sure you have Python and Jupyter Notebook installed
2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Open `AQI_Prediction_Project.ipynb` in Jupyter
4. Run **Kernel → Restart & Run All**
5. The notebook will generate the dataset, save it as `aqi_dataset.csv`, and run the full pipeline automatically

---

##  Libraries Used

| Library | Version | Purpose |
|---|---|---|
| pandas | latest | Data loading and manipulation |
| numpy | latest | Numerical computations |
| matplotlib | latest | Data visualisation |
| seaborn | latest | Statistical charts |
| scikit-learn | latest | ML models and evaluation |

---

*Project completed as part of Introduction to Machine Learning course final project.*  
*Team: Udit, Milan, Dipen, Lotus*
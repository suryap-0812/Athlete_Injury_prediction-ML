# 🏃Athlete Injury Prediction

A **Machine Learning project** designed to **predict injuries in athletes** based on demographic, training, recovery, and wellness data.  
The goal is to help coaches, medical teams, and athletes **identify high-risk individuals** early and make **data-driven decisions** to prevent injuries.

---

## Problem Statement
Sports injuries are a major challenge for athletes and teams, impacting **performance, career longevity, and financial costs**.  
Traditional injury prevention methods rely heavily on **subjective judgment**.  
This project uses **Machine Learning** to analyze athlete data and **predict injury likelihood**, enabling **proactive interventions**.

---

## Dataset
The dataset is **synthetically generated** with **1,000 rows and 15 columns**, based on sports science principles.

| **Column Name**        | **Description** |
|------------------------|-----------------|
| `athlete_id`           | Unique identifier for each athlete *(not predictive)* |
| `age`                  | Athlete's age (years) |
| `gender`               | Male / Female |
| `height_cm`            | Athlete's height (cm) |
| `weight_kg`            | Athlete's weight (kg) |
| `sport_type`           | Type of sport *(running, football, basketball)* |
| `training_load`        | Total weekly or session workload |
| `training_intensity`   | Intensity scale (1-10) |
| `recovery_time_hrs`    | Hours of rest since last session |
| `prior_injury_count`   | Number of past injuries |
| `fatigue_level`        | Fatigue scale (1-10) |
| `wellness_score`       | Composite of wellness factors like sleep, stress |
| `external_load`        | External factors (surface hardness, weather) |
| `injury_flag` *(Target)* | 0 = No Injury, 1 = Injury occurred |
| `days_until_injury`    | Days until injury occurred *(only for injured athletes)* |

---

### **Data Ranges**
| Column               | Min | Max |
|----------------------|-----|-----|
| `age`                | 18  | 34  |
| `height_cm`          | 141.5 | 202.0 |
| `weight_kg`          | 29.4  | 112.0 |
| `training_load`      | 50.1 | 299.7 |
| `training_intensity` | 1 | 10 |
| `recovery_time_hrs`  | 4.0 | 12.0 |
| `prior_injury_count` | 0 | 6 |
| `fatigue_level`      | 1 | 10 |
| `wellness_score`     | 1 | 10 |
| `external_load`      | 0.0 | 10.0 |

---

## ML Pipeline

### **1. Data Preprocessing**
- Handle missing values (`days_until_injury` dropped for initial model).
- Encode categorical features (`gender`, `sport_type` → One-Hot Encoding).
- Standardize numerical features using **StandardScaler**.

### **2. Model Training**
- Train-Test Split: 80% training, 20% testing.
- Baseline Model: **Logistic Regression**.
- Advanced Models to be explored: **Random Forest**, **XGBoost**.

### **3. Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve

---

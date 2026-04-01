# 🔥 Forest Fire Weather Index Prediction Using Regression Models

A comprehensive machine learning project demonstrating data cleaning, exploratory data analysis (EDA), and comparative regression modeling on the Algerian Forest Fires dataset.

---

## 📊 Project Overview

This project predicts the **Fire Weather Index (FWI)** using weather and environmental features from the Algerian Forest Fires dataset. FWI is a critical metric for assessing fire danger conditions. By comparing three different regression models, this project demonstrates best practices in machine learning workflows.

### 🎯 Objective
Build and evaluate multiple regression models to accurately predict Fire Weather Index values based on:
- Weather parameters (Temperature, Humidity, Wind Speed, Rain)
- Fire Danger Indices (FFMC, DMC, DC, ISI, BUI)

---

## 📈 Results Summary

| Model | R² Score | MAE | Performance |
|-------|----------|-----|-------------|
| **Linear Regression** | 0.985 | 0.547 | 🏆 Best |
| Ridge Regression | 0.984 | 0.564 | Excellent |
| Lasso Regression | 0.949 | 1.133 | Good |

**Key Finding:** Linear Regression performed best, explaining **98.5% of variance** in FWI predictions!

---

## 📁 Dataset Information

### Source
**Algerian Forest Fires Dataset** - Contains 246 records of weather observations from two regions in Algeria (June-September 2012)

### Features (12 Input Variables)

#### Weather Data:
- **Temperature**: Noon max temperature (22-42°C)
- **RH**: Relative Humidity (21-90%)
- **Ws**: Wind Speed (6-29 km/h)
- **Rain**: Total daily rainfall (0-16.8 mm)

#### FWI Components (Fire Weather Index System):
- **FFMC**: Fine Fuel Moisture Code (28.6-92.5)
- **DMC**: Duff Moisture Code (1.1-65.9)
- **DC**: Drought Code (7-220.4)
- **ISI**: Initial Spread Index (0-18.5)
- **BUI**: Buildup Index (1.1-68)

### Target Variable
- **FWI**: Fire Weather Index (0-31.1) - *What we're predicting!*
- **Classes**: Binary classification (Fire / Not Fire)

### Data Split
- **Training Set**: 75% (182 samples)
- **Test Set**: 25% (61 samples)

---

## 🛠️ Project Workflow

### 1️⃣ **Data Cleaning & Preprocessing**
- ✅ Loaded Algerian forest fires dataset
- ✅ Identified and handled missing values
- ✅ Created Region column (Bejaia vs Sidi-Bel Abbes)
- ✅ Fixed data types (converted to int/float)
- ✅ Removed extra whitespace from column names
- ✅ Saved cleaned dataset for reproducibility

**Output**: `Cleaned_Dataset.csv`

### 2️⃣ **Exploratory Data Analysis (EDA)**
- 📊 Distribution analysis of all features
- 🔗 Correlation analysis (Pearson correlation)
- 🎯 Feature-target relationships
- 🗓️ Fire distribution by month and region
- 📦 Outlier detection using box plots

**Key Insights**:
- Fire incidents peak in August-September
- Strong correlations between FWI and FFMC, DMC, DC indices
- Different patterns between two regions

### 3️⃣ **Feature Engineering**
- 🗑️ Removed temporal features (day, month, year) - not useful for prediction
- 🔍 **Multicollinearity Detection**: Removed highly correlated features (>0.85 correlation)
  - Features removed help prevent model instability
- 📏 **Standardization**: Applied StandardScaler to normalize features
  - All features scaled to mean=0, std=1
  - Improves model convergence and performance

### 4️⃣ **Model Training & Evaluation**

#### **Model 1: Linear Regression**
```python
Simple linear relationship: y = mx + b
- Fastest to train
- Most interpretable
- Good baseline model
- Result: R² = 0.985 ✅ BEST
```

#### **Model 2: Ridge Regression**
```python
Adds L2 penalty to reduce large coefficients
- Handles multicollinearity better
- Prevents overfitting
- Keeps all features but shrinks their importance
- Result: R² = 0.984 (very close to Linear!)
```

#### **Model 3: Lasso Regression**
```python
Adds L1 penalty with feature selection capability
- Can shrink coefficients to exactly zero
- Good for automatic feature elimination
- More aggressive than Ridge
- Result: R² = 0.949 (good but lower)
```

### 5️⃣ **Evaluation Metrics**

**R² Score (Coefficient of Determination)**
- Measures how well predictions fit actual values
- Range: 0 to 1 (1 = perfect prediction)
- Our best model: 0.985 → explains 98.5% of variance

**MAE (Mean Absolute Error)**
- Average absolute difference between predicted and actual values
- Lower is better
- Our best model: 0.547 → average error of 0.547 FWI units

---

## 💻 Technologies & Libraries Used

```python
# Data Manipulation
- pandas       # Data cleaning and preprocessing
- numpy        # Numerical computing

# Visualization
- matplotlib   # Static plotting
- seaborn      # Statistical data visualization

# Machine Learning
- scikit-learn # ML algorithms and metrics
  ├── LinearRegression
  ├── Ridge
  ├── Lasso
  ├── train_test_split
  ├── StandardScaler
  └── mean_absolute_error, r2_score
```

---

## 🚀 How to Use This Project

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Download `prj_commented.ipynb` from this repository
2. Open in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook prj_commented.ipynb
   ```
3. Run cells sequentially (top to bottom)
4. View outputs and visualizations

### Files in This Project
- `prj_commented.ipynb` - Full annotated notebook with detailed comments
- `Algerian_forest_fires_dataset_UPDATE.csv` - Original dataset
- `Cleaned_Dataset.csv` - Cleaned data (generated during execution)
- `README.md` - This file

---

## 📚 Learning Outcomes

By studying this project, you'll learn:

✅ **Data Cleaning Techniques**
- Handling missing values
- Type conversion and validation
- String manipulation and whitespace handling

✅ **Exploratory Data Analysis**
- Statistical summaries and distributions
- Correlation analysis and visualization
- Feature-target relationships

✅ **Feature Engineering**
- Multicollinearity detection and removal
- Feature scaling and normalization
- Why preprocessing matters

✅ **Regression Modeling**
- Linear Regression fundamentals
- Ridge and Lasso regularization
- Model comparison and selection
- When to use each model type

✅ **ML Best Practices**
- Train-test splitting for unbiased evaluation
- Reproducibility with random_state
- Proper evaluation metrics
- Visualizing model performance

---

## 🔍 Key Insights & Takeaways

### 1. **Simpler Models Can Be Better**
   - Linear Regression beat more complex Lasso
   - Don't always need fancy algorithms!

### 2. **Feature Preprocessing Matters**
   - Standardization improved model stability
   - Multicollinearity removal enhanced interpretability

### 3. **Different Models, Different Purposes**
   - **Linear**: Speed & interpretability
   - **Ridge**: Handles correlated features
   - **Lasso**: Automatic feature selection

### 4. **Evaluation Metrics Tell the Story**
   - High R² (0.985) = predictions very close to actual
   - Low MAE (0.547) = small average prediction error

### 5. **Regional Differences Exist**
   - Fire patterns differ between Bejaia and Sidi-Bel Abbes
   - Peak fire season: August-September

---

## 📊 Visualizations Included

The notebook includes:
- 📈 Histograms - Feature distributions
- 🔥 Heatmaps - Correlation matrices
- 📦 Box plots - Outlier detection
- 📊 Count plots - Fire incidents by month/region
- 📉 Scatter plots - Actual vs predicted values
- 📋 Before/after plots - Effect of standardization

---

## 🎓 For Beginners

**New to Machine Learning?** Here's a simplified workflow:

```
1. Get Data
   ↓
2. Clean & Explore It
   ↓
3. Prepare Features (standardize, remove bad ones)
   ↓
4. Split into Train & Test
   ↓
5. Train Different Models
   ↓
6. Evaluate & Compare Results
   ↓
7. Pick the Best Model! 🏆
```

This project walks through each step with detailed comments!

---

## 🔬 Model Selection Guide

**When to use each model:**

| Model | When to Use | Advantages | Disadvantages |
|-------|------------|------------|---------------|
| **Linear Regression** | Baseline, interpretability needed | Fast, simple, interpretable | Assumes linear relationship |
| **Ridge** | Multicollinearity present | Handles correlated features | Slower than Linear |
| **Lasso** | Feature selection needed | Automatic feature elimination | May eliminate useful features |

---

## 📝 Project Workflow Diagram

```
┌─────────────────────────────────────┐
│   Algerian Forest Fires Dataset     │ 246 records
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│   Data Cleaning & Preprocessing     │ Remove nulls, fix types
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│   Exploratory Data Analysis (EDA)   │ Correlations, distributions
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│   Feature Engineering               │ Scaling, remove multicollinearity
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│   Train-Test Split                  │ 75% train, 25% test
└────────────┬────────────────────────┘
             │
             ├──→ Linear Regression  → R²=0.985 ✅ BEST
             │
             ├──→ Ridge Regression   → R²=0.984
             │
             └──→ Lasso Regression   → R²=0.949
             
             ↓
┌─────────────────────────────────────┐
│   Model Evaluation & Comparison     │ Choose Linear Regression
└─────────────────────────────────────┘
```



### Last Updated: 2024
### Status: ✅ Complete & Documented

OziAgro Farmer-Centred Climate Decision-Support System for Rainfed Rice Production
Working Prototype Documentation - Sections 1-10

1. Project Overview
1.1 Project Title
OziAgro Farmer-Centred Climate Decision-Support System for Rainfed Rice Production
Initial prototype location: Bende Local Government Area, Abia State, Nigeria.

1.2 Purpose
The OziAgro prototype is designed to estimate the short-term risk of a dry spell affecting rainfed rice production and translate that risk into a practical farmer-oriented recommendation.
The system addresses the following decision problem: Given rainfall conditions at a prediction origin, is there an elevated probability of a dry spell of at least seven consecutive dry days occurring within the next 14 days, and what planting action should be considered based on irrigation availability?

1.3 Core Workflow
Historical daily rainfall
        ↓
Data validation
        ↓
Prediction-period definition
        ↓
Dry-day and dry-spell target construction
        ↓
Rainfall feature engineering
        ↓
Machine-learning model
        ↓
Predicted dry-spell probability
        ↓
0.45 decision threshold
        ↓
Low / High risk
        ↓
Irrigation availability
        ↓
Farmer-oriented recommendation

1.4 Prototype Scope
The current prototype is restricted to:
- Rainfed rice production
- Bende LGA
- Short-term dry-spell risk
- Dry day: rainfall <1.0 mm
- Dry spell: ≥7 consecutive dry days
- Prediction horizon: 14 days
- Modelling period: DOY 60 onward
- Historical rainfall: 1981–2025
- XGBoost as the operational prediction model
- Decision threshold: 0.45
- Irrigation availability as the DSS decision input

1.5 Design Principle
The machine-learning model predicts climate risk; the DSS translates that risk into a practical decision.

2. System Objectives and Functional Requirements

2.1 General Objective
To develop a farmer-centred climate decision-support system that estimates short-term dry-spell risk for rainfed rice production and translates the predicted risk into an irrigation-aware planting recommendation.

2.2 Specific Objectives
The prototype is designed to:
- Process and validate historical daily rainfall data for Bende LGA.
- Identify rainfall conditions associated with short-term dry-spell risk.
- Construct rainfall-derived predictors.
- Develop and evaluate machine-learning models for dry-spell prediction.
- Select an operational probability threshold.
- Integrate predicted risk with irrigation availability.
- Provide a reproducible modular decision-support workflow

2.3 Functional Requirements
The system must:
- Load and validate rainfall data.
- Define the modelling period.
- Identify dry days.
- Construct the future dry-spell target.
- Generate rainfall-based predictors.
- Generate a dry-spell probability.
- Classify the probability into Low or High risk.
- Incorporate irrigation availability.
- Produce a farmer-oriented recommendation.

Key temporal definitions
| Parameter                | Definition              |
| ------------------------ | ----------------------- |
| Model start              | DOY 60                  |
| Valid prediction origins | DOY 60–309              |
| Historical cessation     | Approximately DOY 323   |
| Prediction horizon       | 14 days                 |
| Dry day                  | <1.0 mm                 |
| Dry spell                | ≥7 consecutive dry days |
| DSS threshold            | 0.45                    |

3. Data and Study Area

3.1 Study Area
The prototype is initially developed for Bende Local Government Area, Abia State, Nigeria.
The Bende administrative boundary was obtained from the FAO GAUL 2015 Level 2 dataset through Google Earth Engine. The boundary has an approximate area of 603.924 km², with a centroid of approximately 5.655223° N, 7.588168° E.

3.2 Rainfall Dataset
The historical rainfall dataset was derived from CHIRPS v2.0 daily rainfall, with approximately 0.05° spatial resolution (~5 km).
The rainfall data cover 1 January 1981 – 31 December 2025. Daily rainfall within the Bende boundary was aggregated using an area-mean approach to obtain daily rainfall representing the study area. The resulting dataset contains 16,436 daily observations.

3.3 Data Structure
The working dataset contains:
- date
- year
- month
- day
- doy
- rain_mm

3.4 Data Quality
The historical dataset was checked for:
- Missing dates
- Duplicate dates
- Missing rainfall values
- Negative rainfall values
No missing dates, duplicate dates, missing rainfall values, or negative rainfall observations were identified.

3.5 Role of the Dataset
The 1981–2025 dataset is the historical model-development and prototype-testing dataset.

4. Climate-Risk Definitions and Target Construction

4.1 Dry Day
A dry day is defined as Rainfall < 1.0 mm 
Binary classification: Dry day = 1 → rainfall < 1.0 mm
                       Dry day = 0 → rainfall ≥ 1.0 mm

4.2 Dry Spell
A dry spell is defined as seven or more consecutive dry days, with each day receiving less than 1.0 mm rainfall.

4.3 Prediction Horizon
For a prediction origin t the system examines: t+1→t+14
Thus, the prediction horizon is 14 days.

4.4 Modelling Period
The modelling period begins at DOY 60 (1 March). DOY 60 is a modelling boundary and does not represent the observed rainfall onset date.
Valid prediction origins extend from:DOY 60–309.
This leaves 14 subsequent days before the approximate historical cessation point around DOY 323.

4.5 Machine-Learning Target
The target variable is:
Yt = {if a ≥7-day dry spell occurs within the next 14 days}
     {0, otherwise}
Therefore, the model predicts the probability that a dry spell of at least seven consecutive days will occur within the following 14 days.

5. Feature Engineering

Feature engineering converts historical rainfall into predictors representing recent rainfall and dry-condition states.

5.1 Current Dry-Spell State
The system calculates: consecutive_dry_days
This represents the number of consecutive dry days up to the prediction origin.

5.2 Cumulative Rainfall
Rainfall totals are calculated for the preceding:
- 1 day
- 3 days
- 5 days
- 7 days
- 14 days
- 30 days

5.3 Rainy- and Dry-Day Counts
The number of rainy and dry days is calculated over:
- 7 days
- 14 days
- 30 days

Rainy day: Rainfall > 1.0 mm
Dry day: Rainfall < 1.0 mm 

5.4 Seasonal Timing
Two cyclical seasonal variables are generated:
- doy_sin
- doy_cos
These represent seasonal position without treating the end and beginning of the year as unrelated points.

5.5 Final Predictor Set
The model uses 15 predictors:
| Group                 | Predictors                                                                    |
| ----------------------| ----------------------------------------------------------------------------- |
| Dry-state             | consecutive_dry_days                                                          |
| Rainfall accumulation | rainfall_1d, rainfall_3d, rainfall_5d, rainfall_7d,rainfall_14d, rainfall_30d |
| Rainy/dry counts      | rainy_days_7d, dry_days_7d, rainy_days_14d, dry_days_14d, rainy_days_30d, dry_days_30d|
| Seasonal timing       | doy_sin, doy_cos                                                              |
No future rainfall information is used as a predictor.

6. Model Development and Training
6.1 Modelling Dataset
The modelling dataset consists of valid prediction observations with complete predictors and target values.
The 15 rainfall-derived predictors are used to estimate the binary future_dry_spell target.

6.2 Chronological Data Split
A chronological split was adopted to preserve the temporal structure of the rainfall data.
| Dataset          | Period    |
| ---------------- | --------- |
| Training         | 1981–2014 |
| Validation       | 2015–2019 |
| Independent test | 2020–2025 |
The validation dataset was used for model assessment and decision-threshold selection. The independent test dataset was reserved for final evaluation.

6.3 Candidate Models

Three classification models were developed:
Logistic Regression
Random Forest
XGBoost

6.4 XGBoost Configuration
The prototype XGBoost model uses:
| Parameter          |           Value |
| ------------------ | --------------: |
| Estimators         |             300 |
| Maximum depth      |               4 |
| Learning rate      |            0.05 |
| Subsample          |             0.8 |
| Column subsampling |             0.8 |
| Objective          | Binary logistic |
| Evaluation metric  |        Log-loss |
| Random state       |              42 |
Class weighting was implemented using scale_pos_weight to account for the relatively infrequent dry-spell events.

6.5 Operational Model

The trained XGBoost model forms the prediction component of the current OziAgro prototype.

7. Model Evaluation

Model performance is assessed using both discrimination and decision-oriented metrics.

7.1 Evaluation Metrics
The prototype evaluates:
- ROC-AUC
- PR-AUC
- Brier score
- Precision
- Recall
- Specificity
- F1-score
- False-plant rate
- False-wait rate
- Confusion matrix

7.2 Decision-Oriented Metrics
Two metrics are particularly relevant to the DSS:
False-plant rate
The proportion of non-event observations incorrectly classified as high risk.
False-wait rate
The proportion of actual dry-spell events incorrectly classified as low risk.
These metrics help assess the consequences of converting model probabilities into planting decisions.

7.3 Independent Test Evaluation
Using the locked 0.45 threshold, the XGBoost model produced:
| Metric           | Test result |
| ---------------- | ----------: |
| Precision        |      0.1172 |
| Recall           |      0.2679 |
| Specificity      |      0.9217 |
| F1-score         |      0.1630 |
| False-plant rate |      0.0783 |
| False-wait rate  |      0.7321 |
| TN               |       1,331 |
| FP               |         113 |
| FN               |          41 |
| TP               |          15 |
The independent test set was kept separate from threshold selection.

8. Decision Threshold Selection

8.1 Purpose
The model produces a probability between 0 and 1. A threshold is required to convert this probability into an operational risk classification.

Predicted probability
        ↓
Decision threshold
        ↓
Low risk / High risk

8.2 Threshold Selection Rule
The operational threshold was selected using the validation dataset.
The selection rule was, identify thresholds with a false-plant rate ≤ 10%; among eligible thresholds, select the threshold with the highest recall, using F1-score as the tie-breaker.

8.3 Selected Threshold
The selected threshold was: 0.45

At the validation stage, the selected threshold produced:
- Precision: 0.0841
- Recall: 0.2571
- Specificity: 0.9193
- F1-score: 0.1268
- False-plant rate: 0.0807
- False-wait rate: 0.7429

8.4 Final Operational Rule
The prototype therefore uses:
- Probability < 0.45
          → Low risk

- Probability ≥ 0.45
          → High risk
The threshold was selected using validation data and subsequently evaluated on the independent test dataset. The test results were not used to retune the threshold.

9. OziAgro DSS Decision Logic

The DSS combines the model's risk classification with the farmer's irrigation availability.

9.1 Decision Matrix
| Risk | Irrigation | Recommendation                                        |
| ---- | ---------- | ----------------------------------------------------- |
| Low  | No         | Planting can proceed.                                 |
| Low  | Yes        | Planting can proceed; irrigation can serve as backup. |
| High | Yes        | Planting may proceed with supplemental irrigation.    |
| High | No         | Delay planting until dry-spell risk decreases.        |

9.2 Decision Sequence
Predicted probability
        ↓
0.45 threshold
        ↓
Risk classification
        ↓
Irrigation availability
        ↓
Planting recommendation
The DSS therefore does not simply report a model probability. It translates the prediction into a context-dependent action.

10. Prediction Pipeline and Software Architecture

The prototype is implemented as a modular Python system.

10.1 Core Pipeline
Bende rainfall dataset
        ↓
data_loader.py
        ↓
preprocessing.py
        ↓
target_creation.py
        ↓
feature_engineering.py
        ↓
model_training.py
        ↓
trained XGBoost model
        ↓
prediction_service.py
        ↓
dss_logic.py
        ↓
risk + recommendation

10.2 Main Modules
| Module                   | Function                                                 |
| ------------------------ | -------------------------------------------------------- |
| `data_loader.py`         | Loads and validates rainfall data                        |
| `preprocessing.py`       | Defines modelling and prediction periods                 |
| `target_creation.py`     | Creates and verifies the dry-spell target                |
| `feature_engineering.py` | Generates rainfall predictors                            |
| `model_training.py`      | Prepares data, trains and evaluates models               |
| `model_evaluation.py`    | Supports model/result analysis                           |
| `prediction_service.py`  | Generates the operational model prediction               |
| `dss_logic.py`           | Converts risk and irrigation status into recommendations |
| `predict.py`             | Runs the complete prediction workflow                    |
| `app.py`                 | Provides the Streamlit user interface                    |

10.3 Current Operational Prototype

The complete prototype can therefore be represented as:
Historical Bende rainfall
        ↓
Rainfall validation
        ↓
Dry-spell target
        ↓
15 rainfall-derived predictors
        ↓
Trained XGBoost
        ↓
Dry-spell probability
        ↓
0.45 threshold
        ↓
Low / High risk
        ↓
Irrigation availability
        ↓
OziAgro recommendation





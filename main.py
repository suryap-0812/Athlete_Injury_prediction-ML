import pandas as pd
import numpy as np
import json

df = pd.read_csv("assets/athlete_injury_prediction_dataset.csv")

# print("Head:", df.head())
# print("Info:", df.info())
# print("Description:", df.describe())

## Missing values handling

df_mdl = df.drop(columns = ["athlete_id", "days_until_injury"])

#check missing values
print("Missing values:\n", df_mdl.isnull().sum())

df_mdl = pd.get_dummies(df_mdl, columns=['gender', 'sport_type'], drop_first=True)

X = df_mdl.drop(columns=['injury_flag'])
y = df_mdl['injury_flag']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import additional libraries for ML models and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print(" ATHLETE INJURY PREDICTION MODEL ")
print("="*50)

# Model Training and Evaluation
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

model_results = {}

# ==================== Dimensionality Reduction (educational) ====================
try:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif

    print("\n DIMENSIONALITY REDUCTION (educational)\n" + "-"*50)

    # PCA: find components that explain ~95% variance
    pca_full = PCA()
    pca_full.fit(X_train_scaled)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumsum_var >= 0.95) + 1) if np.any(cumsum_var >= 0.95) else min(X_train_scaled.shape[1], 6)

    print(f"Original features: {X_train_scaled.shape[1]}")
    print(f"Components for 95% variance (approx): {n_components}")

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Feature selection: SelectKBest (ANOVA F-test)
    k = min(8, X_train_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = X.columns[selector.get_support()].tolist()
    selected_scores = selector.scores_[selector.get_support()]

    print("Selected features (SelectKBest):")
    for f, s in zip(selected_features, selected_scores):
        print(f"  {f:<25} {s:.2f}")

    # Compare model performance on original vs PCA vs Selected (educational)
    print("\n Comparing models on Original / PCA / Selected feature sets")
    comparison_sets = {
        'Original': (X_train_scaled, X_test_scaled),
        f'PCA_{n_components}': (X_train_pca, X_test_pca),
        f'Selected_{len(selected_features)}': (X_train_selected, X_test_selected)
    }

    comparison_results = {}
    for set_name, (Xt, Xv) in comparison_sets.items():
        print(f"\n-- {set_name} --")
        set_res = {}
        for mname, m in models.items():
            # create a fresh instance of the model to avoid state carryover
            model_copy = type(m)(**m.get_params())
            model_copy.fit(Xt, y_train)
            ypred = model_copy.predict(Xv)
            # Some models may not implement predict_proba; guard accordingly
            try:
                yproba = model_copy.predict_proba(Xv)[:, 1]
            except Exception:
                # fallback: decision_function mapped to probabilities via sigmoid-ish approach
                try:
                    scores = model_copy.decision_function(Xv)
                    # simple scaling to [0,1]
                    yproba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                except Exception:
                    yproba = np.zeros_like(ypred, dtype=float)

            acc = accuracy_score(y_test, ypred)
            # Use roc_auc_score only if we have variation in yproba
            try:
                auc = roc_auc_score(y_test, yproba)
            except Exception:
                auc = float('nan')

            set_res[mname] = {'accuracy': acc, 'roc_auc': auc}
            print(f"{mname:<20} Accuracy: {acc:.4f}, ROC-AUC: {auc if not np.isnan(auc) else 'N/A'}")
        comparison_results[set_name] = set_res

    # Save a simple summary plot
    try:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(1, 1, figsize=(8, 5))
        labels = list(comparison_sets.keys())
        # For plotting, handle missing AUCs by substituting 0
        rf_acc = [comparison_results[l].get('Random Forest', {}).get('accuracy', 0) for l in labels]
        lr_acc = [comparison_results[l].get('Logistic Regression', {}).get('accuracy', 0) for l in labels]
        svm_acc = [comparison_results[l].get('SVM', {}).get('accuracy', 0) for l in labels]
        x = np.arange(len(labels))
        width = 0.25
        ax.bar(x - width, rf_acc, width, label='Random Forest')
        ax.bar(x, lr_acc, width, label='Logistic Regression')
        ax.bar(x + width, svm_acc, width, label='SVM')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy: Original vs PCA vs Selected')
        ax.legend()
        _plt.tight_layout()
        _plt.savefig('dimensionality_reduction_analysis.png', dpi=200)
        _plt.close(fig)
        print("Saved dimensionality reduction analysis: dimensionality_reduction_analysis.png")
    except Exception as e:
        print("Could not save dimensionality reduction plot:", e)

except Exception as e:
    print("Dimensionality reduction step skipped due to error:", e)

# ==================== End Dimensionality Reduction ====================

print("\n Training and Evaluating Models...")
print("-" * 40)

for name, model in models.items():
    print(f"\n Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f" {name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

# Find best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc'])
best_model = model_results[best_model_name]['model']

print(f"\n Best Model: {best_model_name}")
print(f"   Accuracy: {model_results[best_model_name]['accuracy']:.4f}")
print(f"   ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f}")

# Detailed evaluation for best model
print(f"\n Detailed Evaluation for {best_model_name}:")
print("-" * 40)
print(classification_report(y_test, model_results[best_model_name]['y_pred']))

# Feature Importance (for Random Forest)
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n Feature Importance ({best_model_name}):")
    print("-" * 40)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")

# Hyperparameter Tuning for Best Model
print(f"\n Hyperparameter Tuning for {best_model_name}...")
print("-" * 40)

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }
else:  # SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }

grid_search = GridSearchCV(
    models[best_model_name], 
    param_grid, 
    cv=5, 
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
tuned_model = grid_search.best_estimator_

# Evaluate tuned model
y_pred_tuned = tuned_model.predict(X_test_scaled)
y_pred_proba_tuned = tuned_model.predict_proba(X_test_scaled)[:, 1]
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_roc_auc = roc_auc_score(y_test, y_pred_proba_tuned)

print(f" Tuned {best_model_name}:")
print(f"   Best Parameters: {grid_search.best_params_}")
print(f"   Improved Accuracy: {tuned_accuracy:.4f} (vs {model_results[best_model_name]['accuracy']:.4f})")
print(f"   Improved ROC-AUC: {tuned_roc_auc:.4f} (vs {model_results[best_model_name]['roc_auc']:.4f})")

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison
model_names = list(model_results.keys())
accuracies = [model_results[name]['accuracy'] for name in model_names]
roc_aucs = [model_results[name]['roc_auc'] for name in model_names]

axes[0, 0].bar(model_names, accuracies, alpha=0.7, color='skyblue')
axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 2. ROC Curves
for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, model_results[name]['y_pred_proba'])
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {model_results[name]['roc_auc']:.3f})")

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix for Best Model
cm = confusion_matrix(y_test, model_results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 4. Feature Importance (if Random Forest is best)
if best_model_name == 'Random Forest':
    top_features = feature_importance.head(10)
    axes[1, 1].barh(top_features['feature'], top_features['importance'], color='lightgreen')
    axes[1, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Importance')
else:
    # Cross-validation scores
    cv_means = [model_results[name]['cv_mean'] for name in model_names]
    cv_stds = [model_results[name]['cv_std'] for name in model_names]
    
    axes[1, 1].bar(model_names, cv_means, yerr=cv_stds, alpha=0.7, color='lightcoral', capsize=5)
    axes[1, 1].set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('CV Accuracy')
    axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('athlete_injury_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the best tuned model
joblib.dump(tuned_model, 'best_athlete_injury_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\n Model and Scaler Saved!")
print("   - best_athlete_injury_model.pkl")
print("   - scaler.pkl")
print("   - athlete_injury_prediction_results.png")

# Print feature information for debugging
print(f"\n Feature Information:")
print(f"   Number of features: {X.shape[1]}")
print(f"   Feature names: {list(X.columns)}")

# Example prediction function
def predict_injury_risk(input_data):
    """
    Predict injury probability for a new athlete
    input_data should be a dictionary with all required features
    """
    # Create DataFrame with the same structure as training data
    input_df = pd.DataFrame([input_data])
    
    # Apply the same preprocessing as training data
    if 'gender' in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=['gender'], prefix='gender')
    if 'sport_type' in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=['sport_type'], prefix='sport_type')
    
    # Ensure all columns from training are present
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data             
    input_df = input_df[X.columns]
    
    # Scale features
    features_scaled = scaler.transform(input_df)
    
    # Make prediction
    probability = tuned_model.predict_proba(features_scaled)[0][1]
    prediction = tuned_model.predict(features_scaled)[0]
    
    return prediction, probability

print("\n Example Prediction:")
print("-" * 40)
# Example: 25-year-old male, 180cm, 75kg, basketball player
example_data = {
    'age': 25,
    'height_cm': 180,
    'weight_kg': 75,
    'training_load': 8,
    'training_intensity': 7,
    'recovery_time_hrs': 6,
    'prior_injury_count': 1,
    'fatigue_level': 6,
    'wellness_score': 7,
    'external_load': 850,
    'gender': 'male',
    'sport_type': 'basketball'
}

try:
    example_pred, example_prob = predict_injury_risk(example_data)
    print(f"Athlete Profile: {example_data['age']}yr old {example_data['gender']} {example_data['sport_type']} player")
    print(f"Injury Prediction: {'  HIGH RISK' if example_pred == 1 else ' LOW RISK'}")
    print(f"Injury Probability: {example_prob:.2%}")
except Exception as e:
    print(f"Prediction error: {e}")
    print("Available features for prediction:", list(X.columns))

print(f"Injury Prediction: {'High Risk' if example_pred == 1 else 'Low Risk'}")
print(f"Injury Probability: {example_prob:.2%}")

print("\n" + "="*50)
print(" ANALYSIS COMPLETE! ")
print("="*50)
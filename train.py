import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
import joblib

# Create assets folder for plots
os.makedirs('assets', exist_ok=True)

print("1. Loading dataset...")
df = pd.read_csv('emails.csv')

# Drop non-feature and label columns
X = df.drop(columns=['Email No.', 'Prediction'])
y = df['Prediction']

feature_words = list(X.columns)

# Save feature words list to JSON
with open('feature_words.json', 'w') as f:
    json.dump(feature_words, f)
print(f"Saved {len(feature_words)} feature words to feature_words.json")

# 2. Train-Test Split (Stratified to maintain class balance)
print("2. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Define models to train
models = {
    'Naive Bayes (Multinomial)': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (Linear)': SVC(kernel='linear', probability=True, random_state=42)
}

results = {}

print("3. Training and evaluating models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"{name} Evaluation Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")

# 4. Find the best model based on F1-score
best_model_name = max(results, key=lambda k: results[k]['f1'])
best_model_info = results[best_model_name]
print(f"\nBest Model: {best_model_name} with F1-Score of {best_model_info['f1']:.4f}")

# Save the best model
joblib.dump(best_model_info['model'], 'spam_classifier.joblib')
print(f"Saved the best model to spam_classifier.joblib")

# 5. Visualizing Evaluation metrics
print("\n4. Generating plots...")
# Plot metrics comparison
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()]
})

metrics_melted = pd.melt(metrics_df, id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, palette='viridis')
plt.title('Spam Classifier Models Comparison')
plt.ylim(0.8, 1.02)
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('assets/model_comparison.png', dpi=300)
plt.close()

# Plot Confusion Matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['Ham', 'Spam'])
    ax.set_yticklabels(['Ham', 'Spam'])
plt.tight_layout()
plt.savefig('assets/confusion_matrices.png', dpi=300)
plt.close()

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/roc_curves.png', dpi=300)
plt.close()

# 6. Extract Top Words Associated with Spam using Naive Bayes
mnb_model = results['Naive Bayes (Multinomial)']['model']
# Log ratio of features: log P(Word | Spam) - log P(Word | Ham)
word_scores = mnb_model.feature_log_prob_[1] - mnb_model.feature_log_prob_[0]
top_spam_indices = np.argsort(word_scores)[::-1][:20]
top_spam_words = [feature_words[i] for i in top_spam_indices]
top_spam_scores = [word_scores[i] for i in top_spam_indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_spam_scores, y=top_spam_words, palette='rocket')
plt.title('Top 20 Words Indicating Spam (Naive Bayes Log Likelihood Ratio)')
plt.xlabel('Log Likelihood Ratio (Spam vs Ham)')
plt.ylabel('Words')
plt.tight_layout()
plt.savefig('assets/top_spam_words.png', dpi=300)
plt.close()

print("\nAll plots generated and saved to the 'assets/' folder:")
print("- assets/model_comparison.png")
print("- assets/confusion_matrices.png")
print("- assets/roc_curves.png")
print("- assets/top_spam_words.png")
print("\nTraining completed successfully!")

import sys
import re
import json
import pandas as pd
from collections import Counter
import joblib

def load_resources():
    try:
        model = joblib.load('spam_classifier.joblib')
        with open('feature_words.json', 'r') as f:
            feature_words = json.load(f)
        return model, feature_words
    except Exception as e:
        print(f"Error loading model resources: {e}")
        print("Please run train.py first to train the model and save resources.")
        sys.exit(1)

def extract_features(text, feature_words):
    # Convert text to lowercase and extract word tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    token_counts = Counter(tokens)
    
    # Construct the 3000-dimensional feature vector
    vector = [token_counts.get(word, 0) for word in feature_words]
    # Return as DataFrame with correct feature names to avoid scikit-learn warnings
    df_vector = pd.DataFrame([vector], columns=feature_words)
    return df_vector, token_counts


def analyze_prediction(model, token_counts, feature_words):
    # For Logistic Regression, coefficients represent the weight of each word
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        contributions = []
        
        for word, count in token_counts.items():
            if word in feature_words:
                idx = feature_words.index(word)
                weight = coefficients[idx]
                contribution = count * weight
                contributions.append((word, count, weight, contribution))
        
        # Sort by contribution (positive = points to Spam, negative = points to Ham)
        contributions.sort(key=lambda x: x[3], reverse=True)
        return contributions
    return None

def main():
    print("=" * 60)
    print("           SPAM EMAIL CLASSIFIER - INTERACTIVE CLI           ")
    print("=" * 60)
    print("Loading model and vocabulary...")
    model, feature_words = load_resources()
    print("Model loaded successfully!")
    print("=" * 60)
    
    while True:
        print("\nEnter/paste your email content below. (Press Enter on an empty line when finished):")
        
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                if not lines:
                    print("\nExiting. Goodbye!")
                    return
                break
        
        email_text = "\n".join(lines).strip()
        
        if not email_text:
            print("No email content entered. Please try again.")
            continue
            
        # Extract features
        vector, token_counts = extract_features(email_text, feature_words)
        
        # Predict
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        
        spam_prob = probabilities[1]
        ham_prob = probabilities[0]
        
        print("\n" + "-" * 40)
        print("                 RESULT                 ")
        print("-" * 40)
        
        if prediction == 1:
            print(f"Prediction: ** SPAM **")
            print(f"Confidence: {spam_prob * 100:.2f}%")
        else:
            print(f"Prediction: ** HAM (Legitimate Email) **")
            print(f"Confidence: {ham_prob * 100:.2f}%")
            
        # Analyze contribution of words
        contributions = analyze_prediction(model, token_counts, feature_words)
        if contributions:
            print("\nKey Contributing Words:")
            if prediction == 1:
                # Show top spam indicators found in email
                spam_indicators = [c for c in contributions if c[3] > 0][:5]
                if spam_indicators:
                    print("  [Spam Indicators]")
                    for word, count, weight, contrib in spam_indicators:
                        print(f"    - '{word}' (count: {count}, impact: +{contrib:.2f})")
                else:
                    print("  No strong positive spam word indicators found.")
            else:
                # Show top ham indicators found in email
                ham_indicators = sorted([c for c in contributions if c[3] < 0], key=lambda x: x[3])[:5]
                if ham_indicators:
                    print("  [Ham/Legitimate Indicators]")
                    for word, count, weight, contrib in ham_indicators:
                        print(f"    - '{word}' (count: {count}, impact: {contrib:.2f})")
                else:
                    print("  No strong legitimate word indicators found.")
        
        print("-" * 40)
        
        # Ask to continue
        try:
            choice = input("\nClassify another email? (y/n, default='y'): ").strip().lower()
            if choice in ['n', 'no']:
                print("Goodbye!")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()


from helpers.helper_cleaners import get_cleaned_data
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 

# 1. Fetch and prepare data
data = get_cleaned_data()
X = data['text']
y = data['polarity']

# 2. Split data into training and testing sets (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 3. Build the ML pipeline (Vectorization -> Classification)
pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()), 
    ('classifier', LinearSVC())
])

# 4. Train and evaluate with default parameters
accuracy_1 = pipe.fit(X_train, y_train).score(X_test, y_test)
print(f"First accuracy (Default): {accuracy_1 * 100:.2f}%")

# 5. Hyperparameter tuning (adjusting C parameter) and re-evaluating
accuracy_2 = pipe.set_params(classifier__C=10).fit(X_train, y_train).score(X_test, y_test)
print(f"Second accuracy (C=10): {accuracy_2 * 100:.2f}%")

# 6. Serialize and save the model for API integration
joblib.dump(pipe, 'sentiment_model.joblib')
print("Model saved successfully as 'sentiment_model.joblib'!")
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def run_all_models():
    # Load dataset with proper encoding
    df = pd.read_csv('data/car_data.csv', encoding='ISO-8859-1')

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=['number'])

    # Define features and target
    X = df_numeric.drop('Car Purchase Amount', axis=1, errors='ignore')
    y = df_numeric['Car Purchase Amount'] > df_numeric['Car Purchase Amount'].mean()  # binary classification

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    }

    # Train and evaluate
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        }

    # Ensure plot directory exists
    os.makedirs('static/plots', exist_ok=True)

    # Plotting all metrics with the same navy blue color for all bars
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    plot_paths = []
    navy_blue = '#001f3f'

    for metric in metric_names:
        plt.figure()
        scores = [metrics[m][metric] for m in models]
        bars = plt.bar(models.keys(), scores, color=navy_blue)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom')
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = f'static/plots/{metric.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)

    return metrics, plot_paths
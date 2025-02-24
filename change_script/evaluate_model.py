# scripts/3_evaluate_model.py
 
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score,
    precision_score,
    recall_score
)
import joblib
import os

def evaluate_cross_dataset():
    # Configure paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_path, "data", "combined_data.csv")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load and prepare data
    try:
        data = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        raise SystemExit("Error: Combined data file not found. Run preprocessing first.")

    # Create target variable if missing
    if 'target' not in data.columns:
        if 'ema_neg_score' not in data.columns:
            raise ValueError("Missing required 'ema_neg_score' column for target creation")
        data['target'] = pd.cut(data['ema_neg_score'],
                              bins=[-np.inf, data['ema_neg_score'].median(), np.inf],
                              labels=[0, 1])

    # Initialize results tracking
    full_results = []
    cluster_reports = []

    # Evaluate each cluster
    for cluster in sorted(data['cluster'].unique()):
        try:
            # Load model
            model_path = os.path.join(base_path, "models", f"model_cluster_{cluster}.pkl")
            if not os.path.exists(model_path):
                print(f"⚠️ Model for cluster {cluster} not found. Skipping.")
                continue
                
            model = joblib.load(model_path)
            
            # Prepare test data
            test_data = data[(data['cluster'] == cluster) & 
                           (data['dataset_source'] == 'crosscheck')]
                           
            if test_data.empty:
                print(f"⏩ No crosscheck data for cluster {cluster}. Skipping.")
                continue

            # Prepare features
            X_test = test_data.select_dtypes(include=np.number).drop(
                columns=['target', 'dataset_source', 'cluster', 'subject_id'], 
                errors='ignore'
            )
            y_test = test_data['target']

            # Validate feature dimensions
            if X_test.shape[1] != model.n_features_in_:
                raise ValueError(f"Feature mismatch: Model expects {model.n_features_in_} features, "
                               f"Test data has {X_test.shape[1]} features")

            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'cluster': cluster,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            full_results.append(metrics)

            # Generate classification report
            report = classification_report(
                y_test, y_pred, 
                target_names=['Low Stress', 'High Stress'],
                output_dict=True,
                zero_division=0
            )
            cluster_reports.append({
                'cluster': cluster,
                **report['weighted avg']
            })

            # Print cluster results
            print(f"\n{'═'*40}")
            print(f"Cluster {cluster} Evaluation Results")
            print(f"{'═'*40}")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"F1 Score: {metrics['f1']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            #print("\n Detailed Classification Report:")
            #print(classification_report(y_test, y_pred, target_names=['Low Stress', 'High Stress']))

        except Exception as e:
            print(f"❌ Error evaluating cluster {cluster}: {str(e)}")
            continue

    # Save aggregated results
    if full_results:
        # Save numeric metrics
        pd.DataFrame(full_results).to_csv(
            os.path.join(results_dir, "cross_dataset_metrics.csv"), 
            index=False
        )
        
        # Save classification reports
        pd.DataFrame(cluster_reports).to_csv(
            os.path.join(results_dir, "classification_reports.csv"),
            index=False
        )
        
        print(f"\nResults saved to {results_dir}/")
    else:
        print("\n⚠️ No evaluation results generated")

if __name__ == "__main__":
    evaluate_cross_dataset()


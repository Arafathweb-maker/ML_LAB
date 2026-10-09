"""
Experiment 1: Decision Tree Classification
Objective: Implement decision trees with the Iris dataset and classify data

Requirements:
1. Load the Iris dataset
2. Explore the dataset
3. Display basic information, missing values and descriptive statistics
4. Create a pair plot of Iris features
5. Split data into training and testing
6. Train a DecisionTreeClassifier
7. Predict test data
8. Display classification report
9. Generate and save confusion matrix image
10. Visualize the trained decision tree
11. Predict a new sample
12. Print final result
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# Create output folder if it doesn't exist
OUTPUT_FOLDER = "output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("=" * 80)
    print("EXPERIMENT 1: DECISION TREE CLASSIFICATION WITH IRIS DATASET")
    print("=" * 80)
    print("\n[STEP 1] Loading the Iris Dataset...")
    
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Create a DataFrame for better exploration
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["Target"] = y
    df["Target_Name"] = df["Target"].map(
        {0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]}
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {iris.target_names}")
    
    return df, X, y, iris


def display_dataset_info(df):
    """Display basic information about the dataset"""
    print("\n" + "=" * 80)
    print("[STEP 2 & 3] Dataset Information")
    print("=" * 80)
    
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nDataset shape:", df.shape)
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() == 0:
        print("✓ No missing values found!")
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nClass Distribution:")
    print(df["Target_Name"].value_counts())


def create_pair_plot(df):
    """Create and save a pair plot of Iris features"""
    print("\n" + "=" * 80)
    print("[STEP 4] Creating Pair Plot...")
    print("=" * 80)
    
    # Select only the feature columns for pair plot
    plot_df = df.copy()
    
    # Create pair plot
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(
        plot_df,
        hue="Target_Name",
        diag_kind="hist",
        vars=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        palette="husl",
    )
    
    plt.suptitle("Pair Plot of Iris Features", y=1.00, fontsize=16, fontweight="bold")
    pair_plot_path = os.path.join(OUTPUT_FOLDER, "pairplot_iris.png")
    pairplot.savefig(pair_plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Pair plot saved to: {pair_plot_path}")
    plt.close()


def split_and_train(X, y, test_size=0.3, random_state=42):
    """Split data and train the Decision Tree Classifier"""
    print("\n" + "=" * 80)
    print("[STEP 5] Splitting Dataset into Training and Testing...")
    print("=" * 80)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set size: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    print("\n[STEP 6] Training Decision Tree Classifier...")
    
    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=random_state, max_depth=5)
    clf.fit(X_train, y_train)
    
    print("✓ Decision Tree trained successfully!")
    print(f"Tree depth: {clf.get_depth()}")
    print(f"Number of leaves: {clf.get_n_leaves()}")
    
    return X_train, X_test, y_train, y_test, clf


def make_predictions(clf, X_test, y_test):
    """Make predictions on test data"""
    print("\n" + "=" * 80)
    print("[STEP 7] Predicting Test Data...")
    print("=" * 80)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Predictions completed!")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return y_pred


def display_classification_report(y_test, y_pred, iris):
    """Display classification report"""
    print("\n" + "=" * 80)
    print("[STEP 8] Classification Report")
    print("=" * 80)
    
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    print("\n" + report)
    
    return report


def generate_confusion_matrix(y_test, y_pred, iris):
    """Generate and save confusion matrix image"""
    print("\n" + "=" * 80)
    print("[STEP 9] Generating Confusion Matrix...")
    print("=" * 80)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a figure and plot confusion matrix using ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=iris.target_names
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    
    plt.title("Confusion Matrix - Decision Tree Iris Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUT_FOLDER, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to: {cm_path}")
    plt.close()


def visualize_decision_tree(clf, iris):
    """Visualize and save the trained decision tree"""
    print("\n" + "=" * 80)
    print("[STEP 10] Visualizing Decision Tree...")
    print("=" * 80)
    
    # Create a figure with larger size for better visualization
    plt.figure(figsize=(25, 15))
    
    plot_tree(
        clf,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    
    tree_path = os.path.join(OUTPUT_FOLDER, "decision_tree.png")
    plt.savefig(tree_path, dpi=300, bbox_inches="tight")
    print(f"✓ Decision tree visualization saved to: {tree_path}")
    plt.close()


def predict_new_sample(clf, iris):
    """Predict class for a new sample"""
    print("\n" + "=" * 80)
    print("[STEP 11-13] Predicting New Sample")
    print("=" * 80)
    
    # New sample: [Sepal Length, Sepal Width, Petal Length, Petal Width]
    new_sample = np.array([[4.8, 2.9, 1.3, 0.2]])
    
    print("\nNew Sample:")
    print(f"  Sepal Length: {new_sample[0][0]} cm")
    print(f"  Sepal Width:  {new_sample[0][1]} cm")
    print(f"  Petal Length: {new_sample[0][2]} cm")
    print(f"  Petal Width:  {new_sample[0][3]} cm")
    
    # Make prediction
    prediction = clf.predict(new_sample)
    prediction_proba = clf.predict_proba(new_sample)
    
    print("\n" + "=" * 80)
    print("FINAL RESULT - NEW SAMPLE PREDICTION")
    print("=" * 80)
    print(f"\nPredicted Class: {iris.target_names[prediction[0]]}")
    print(f"Prediction Index: {prediction[0]}")
    
    print("\nPrediction Probabilities:")
    for i, class_name in enumerate(iris.target_names):
        print(f"  {class_name}: {prediction_proba[0][i]:.4f} ({prediction_proba[0][i]*100:.2f}%)")
    
    return prediction, prediction_proba


def main():
    """Main execution function"""
    try:
        # Step 1: Load and explore data
        df, X, y, iris = load_and_explore_data()
        
        # Step 2-3: Display dataset information
        display_dataset_info(df)
        
        # Step 4: Create pair plot
        create_pair_plot(df)
        
        # Step 5-6: Split data and train model
        X_train, X_test, y_train, y_test, clf = split_and_train(X, y)
        
        # Step 7: Make predictions
        y_pred = make_predictions(clf, X_test, y_test)
        
        # Step 8: Display classification report
        display_classification_report(y_test, y_pred, iris)
        
        # Step 9: Generate confusion matrix
        generate_confusion_matrix(y_test, y_pred, iris)
        
        # Step 10: Visualize decision tree
        visualize_decision_tree(clf, iris)
        
        # Step 11-13: Predict new sample
        predict_new_sample(clf, iris)
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nAll output files saved to the '{OUTPUT_FOLDER}' folder:")
        print("  ✓ pairplot_iris.png")
        print("  ✓ confusion_matrix.png")
        print("  ✓ decision_tree.png")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

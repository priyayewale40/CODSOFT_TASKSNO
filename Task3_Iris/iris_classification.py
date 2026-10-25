"""CodSoft Task 3: Iris Flower Classification."""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=150, random_state=42)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Iris Flower Classification ===")
    print(f"Dataset samples: {len(X)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    example = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
    predicted = model.predict(example)[0]
    print("\nExample measurements:", example[0])
    print("Predicted species:", iris.target_names[predicted])

if __name__ == "__main__":
    main()

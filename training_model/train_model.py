import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SAMPLING_RATE = 100

WINDOW_SIZE = SAMPLING_RATE
STEP_SIZE = WINDOW_SIZE // 2


def magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


def extract_features(window):
    features = []

    sensor_columns = [
        "arm_x",
        "arm_y",
        "arm_z",
        "leg_x",
        "leg_y",
        "leg_z"
    ]

    window["arm_mag"] = magnitude(
        window["arm_x"],
        window["arm_y"],
        window["arm_z"]
    )

    window["leg_mag"] = magnitude(
        window["leg_x"],
        window["leg_y"],
        window["leg_z"]
    )

    all_columns = sensor_columns + ["arm_mag", "leg_mag"]

    for col in all_columns:
        values = window[col].values

        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.max(values) - np.min(values),
            np.sum(values ** 2) / len(values)
        ])

    return features


def load_dataset():
    X = []
    y = []

    required_columns = [
        "timestamp",
        "arm_x",
        "arm_y",
        "arm_z",
        "leg_x",
        "leg_y",
        "leg_z"
    ]

    print("DATA_DIR:", DATA_DIR)
    print("Exists:", os.path.exists(DATA_DIR))
    print("Folders:", os.listdir(DATA_DIR))

    for label in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, label)

        if not os.path.isdir(folder_path):
            continue

        print(f"\nLoading label: {label}")

        for filename in os.listdir(folder_path):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df = df.dropna()

            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing column {col} in file {file_path}")

            if len(df) < WINDOW_SIZE:
                print(f"Skipping {filename}, too few rows")
                continue

            dt = df["timestamp"].diff().mean()

            if dt > 0:
                estimated_hz = 1000000 / dt
                print(f"{filename}: approx {estimated_hz:.1f} Hz, rows: {len(df)}")

            for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
                window = df.iloc[start:start + WINDOW_SIZE].copy()

                features = extract_features(window)

                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)


def main():
    print("Loading dataset...")

    X, y = load_dataset()

    print("\nSamples:", len(X))

    if len(X) == 0:
        raise ValueError("No samples found. Check your data folders and CSV files.")

    print("Features per sample:", X.shape[1])
    print("Labels:", sorted(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\nRandom Forest accuracy:")
    print(accuracy_score(y_test, rf_pred))

    print("\nRandom Forest confusion matrix:")
    print(confusion_matrix(y_test, rf_pred))

    print("\nRandom Forest classification report:")
    print(classification_report(y_test, rf_pred))

    print("\nTraining Decision Tree...")

    tree_model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )

    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)

    print("\nDecision Tree accuracy:")
    print(accuracy_score(y_test, tree_pred))

    print("\nDecision Tree confusion matrix:")
    print(confusion_matrix(y_test, tree_pred))

    print("\nDecision Tree classification report:")
    print(classification_report(y_test, tree_pred))

    joblib.dump(rf_model, os.path.join(BASE_DIR, "random_forest_model.pkl"))
    joblib.dump(tree_model, os.path.join(BASE_DIR, "decision_tree_model.pkl"))

    print("\nModels saved:")
    print("random_forest_model.pkl")
    print("decision_tree_model.pkl")

    print("\nDecision Tree rules:")
    rules = export_text(tree_model)
    print(rules)


if __name__ == "__main__":
    main()
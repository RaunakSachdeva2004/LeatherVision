import cv2
import numpy as np
import os
import argparse
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "leather_model.pkl"


# LBP Feature Extraction
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 1

    lbp = np.zeros_like(gray)

    for i in range(radius, gray.shape[0] - radius):
        for j in range(radius, gray.shape[1] - radius):
            center = gray[i, j]
            binary_string = ''

            for dx, dy in [(-1,-1), (-1,0), (-1,1),
                (0,1), (1,1), (1,0),
                (1,-1), (0,-1)]:
                binary_string += '1' if gray[i+dx, j+dy] > center else '0'

            lbp[i, j] = int(binary_string, 2)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist


# Load Dataset
def load_data(dataset_path):
    data, labels = [], []

    for label, folder in enumerate(["real", "fake"]):
        folder_path = os.path.join(dataset_path, folder)

        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            features = extract_lbp(img)

            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels)


# Train Model
def train(dataset_path):
    print("[INFO] Loading dataset...")
    X, y = load_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n[RESULTS]")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


# Predict
def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Train model first!")
        return

    model = joblib.load(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Invalid image path")
        return

    img = cv2.resize(img, (128, 128))
    features = extract_lbp(img).reshape(1, -1)

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]

    label = "Real Leather" if prediction == 0 else "Fake Leather"

    print("\n[RESULT]")
    print(f"Prediction : {label}")
    print(f"Confidence : {max(confidence):.2f}")


# CLI Setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leather Classifier CLI")

    parser.add_argument("mode", choices=["train", "predict"],
                        help="train or predict")

    parser.add_argument("--dataset", type=str,
                        help="Path to dataset")

    parser.add_argument("--image", type=str,
                        help="Path to image")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.dataset:
            print("Provide dataset path using --dataset")
        else:
            train(args.dataset)

    elif args.mode == "predict":
        if not args.image:
            print("Provide image path using --image")
        else:
            predict(args.image)
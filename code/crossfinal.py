import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score  # ✅ Added missing import

# ----- Load dataset -----
df = pd.read_csv(r"F:\bearing_fault_project\data\feature_time_48k_2048_load_1.csv")

# ----- Features and target -----
X = df.drop("fault", axis=1)
y = df["fault"]

# ----- Label Encoding and Scaling -----
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- K-Fold Cross Validation -----
print("\nPerforming Stratified K-Fold Cross-Validation (k=5)...\n")
svm_clf = SVC(kernel='rbf', gamma='scale')

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_clf, X_scaled, y_encoded, cv=kfold, scoring='accuracy')

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# ----- Train-Test Split for Final Evaluation -----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)

print("\nTest Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report on Test Set:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ----- PCA Visualization -----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=label_encoder.inverse_transform(y_encoded),
    palette="tab10", alpha=0.7
)
plt.title("2D PCA Projection of Bearing Fault Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Fault Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

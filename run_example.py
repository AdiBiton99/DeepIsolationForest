import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

# ייבוא המודל מהפרוייקט
from algorithms.dif import DIF

# 1) יוצרים נתונים דמה: רוב ההדגימות 'נורמליות' + מעט אנומליות
rng = np.random.RandomState(42)
X_normal = rng.normal(loc=0.0, scale=1.0, size=(1000, 10))
X_outliers = rng.uniform(low=8.0, high=10.0, size=(50, 10))
X = np.vstack([X_normal, X_outliers])
y = np.hstack([np.zeros(X_normal.shape[0]), np.ones(X_outliers.shape[0])])  # 1 = אנומליה

# 2) הכנת נתוני אימון ובדיקה
# מאמנים את המודל על דוגמאות רגילות בלבד (unsupervised anomaly detection)
X_train = X[y == 0]
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3) הגדרת ואימון המודל
model_configs = {'n_ensemble': 50, 'n_estimators': 6}
model = DIF(**model_configs, device='cpu')
print("Training on", X_train.shape[0], "normal samples...")
model.fit(X_train)

# 4) חישוב ציון אנומליות על קבוצת הבדיקה
scores = model.decision_function(X_test)  # ציון: ככל שגדול יותר — יותר אנומליה (אם צריך להחליף סימן, ראו הערה)
print("Scores shape:", scores.shape)

# הדפסת 10 ציונים ראשונים לצורך בדיקה
print("First 10 scores:", scores[:10])
print("First 10 true labels:", y_test[:10])

# 5) הערכה (ROC AUC)
try:
    auc = roc_auc_score(y_test, scores)
except ValueError:
    # לפעמים הסימן הפוך — נבדוק גם את המקרה ההפוך
    auc = roc_auc_score(y_test, -scores)
print("ROC AUC:", auc)

# 6) שמירת המודל לדיסק
with open("dif_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model to dif_model.pkl")
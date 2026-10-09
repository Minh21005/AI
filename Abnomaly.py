from sklearn.ensemble import IsolationForest
import numpy as np

n = 200

X = np.random.uniform(-20.0, 80, n)

X = X.reshape(-1, 1)

clf = IsolationForest(random_state=42, contamination=0.05)

clf.fit(X)

prediction = clf.predict(X)

print("Input data:")
print(np.round(X.flatten(), 2))
print("Result(-1: Abnomaly | 1: Normal):")
print(prediction)

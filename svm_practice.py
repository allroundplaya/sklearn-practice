from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
import numpy as np


X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# print(X)
# print(y)
# y = np.where(y==0, 'blue', 'red')
# print(y)
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])



# plt.plot(X[:, 0], X[:, 1], data=y,  fmt='bo')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap("rainbow", 2))

# plt.legend(plt.cm.get_cmap("rainbow", 2))
plt.legend()
plt.colorbar(ticks=range(2), label= "y value")


plt.show()
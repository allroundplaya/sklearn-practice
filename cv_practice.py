from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

# create a synthetic dataset
X, y = make_blobs(random_state= 0)

# split data and labels into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate a model and fit it to the training set
logreg = LogisticRegression().fit(X_train, y_train)


# evaluate the model on the test set
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))

plt.grid(linestyle= "--")

plt.xlabel("x1")
plt.xlabel("x2")
plt.scatter(X[:, 0], X[:, 1], c= y, cmap= plt.cm.get_cmap("rainbow", 3))
plt.colorbar(ticks=range(3), format="group %d", label="color")
plt.show()



# plt.title("practice 1")
# plt.xlabel("x1")
# plt.ylabel("x2")



# plt.scatter(X[:, 0], X[:, 1], c=y,  cmap=plt.cm.Blues)

plt.show()

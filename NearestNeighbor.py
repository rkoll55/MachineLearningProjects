from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# A kaggle project I initially completed on codecademy
# Kaggle link https://www.kaggle.com/competitions/breast-cancer-classification/data

breast_cancer_data = load_breast_cancer()

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target,test_size = 0.2, random_state = 100)

klist = []
accuracies = []
for k in range(1,100):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  klist.append(k)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(klist,accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
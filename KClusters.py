import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import(KMeans)

# A kaggle project I initially completed on codecademy
# Kaggle link https://www.kaggle.com/datasets/landlord/handwriting-recognition


digits = datasets.load_digits()
#print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
#plt.show()
#print(digits.target[100])
model = KMeans(n_clusters =10, random_state = 42)
model.fit(digits.data)

fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.53,2.67,0.00,0.00,0.00,0.00,0.00,2.44,7.24,6.71,0.00,0.00,0.00,0.00,2.21,7.55,6.86,1.53,0.00,0.00,0.00,0.00,6.25,7.62,7.62,7.55,5.95,0.23,0.00,0.00,6.79,7.40,4.65,7.01,7.62,0.61,0.00,0.00,2.44,7.09,7.62,7.62,3.97,0.00,0.00,0.00,0.00,0.38,1.52,1.45,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.46,2.59,0.00,0.00,0.00,0.00,0.00,0.00,2.74,7.62,1.60,0.00,0.00,0.00,0.00,0.00,1.37,7.62,2.97,0.00,0.00,0.00,0.00,0.00,0.77,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.46,7.62,3.66,0.00,0.00,0.00,0.00,0.00,0.00,2.29,0.77,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.51,5.26,0.00,0.00,0.00,0.00,0.00,0.00,7.17,7.63,0.00,0.00,0.00,0.00,0.00,0.00,5.26,7.09,0.00,0.00,0.00,0.00,0.00,1.84,7.55,5.80,5.11,1.30,0.00,0.00,0.00,6.18,7.62,7.62,7.09,1.30,0.00,0.00,0.00,7.63,6.18,3.81,0.76,0.00,0.00,0.00,0.00,0.61,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,2.14,3.81,1.37,0.00,0.00,0.00,0.00,6.48,7.62,7.62,7.32,0.61,0.00,0.00,0.00,4.35,2.90,3.51,7.62,2.52,0.00,0.00,0.00,0.00,0.00,7.01,7.62,7.62,1.22,0.00,0.00,0.00,0.00,3.81,5.65,7.62,1.45,0.00,0.00,0.00,0.23,4.04,7.40,6.10,0.08,0.00,0.00,0.00,0.83,7.17,5.95,0.76,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

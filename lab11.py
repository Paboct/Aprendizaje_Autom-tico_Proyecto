import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, BisectingKMeans,MiniBatchKMeans
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_sample_image
import seaborn as sns

#biblioteca para usar graficos
sns.set()

#genera datos de muestreo
X, y = make_blobs(n_samples=300, centers=4,random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

#aplica k means
est = MiniBatchKMeans(4)
est.fit(X)
y_kmeans = est.predict(X) #tendra
"""
est_mini_batch = MiniBatchKMeans(4)
est_mini_batch.fit(X)
y_mini_batch_kmeans = est_mini_batch.predict(X)
"""

#crea el grafico
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow')
plt.show()

#aplicar Kmeans a datos de digitos y visualizar los datos
digits = load_digits()
est = KMeans(n_clusters=10)
clusters = est.fit_predict(digits.data)

#visualizacion de los clusters
est.cluster_centers_.shape
fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(est.cluster_centers_[i].reshape((8, 8)),
cmap=plt.cm.binary)
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
X = PCA(2).fit_transform(digits.data)
kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
edgecolor='none', alpha=0.6)

#crea graficas para comparar con las etiquetas, es decir, matriz de confusion
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, **kwargs)
ax[0].set_title('learned cluster labels')
ax[1].scatter(X[:, 0], X[:, 1], c=digits.target, **kwargs)
ax[1].set_title('true labels')
plt.show()
print(accuracy_score(digits.target, labels))
print(confusion_matrix(digits.target, labels))
plt.imshow(confusion_matrix(digits.target, labels),
cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()

#carga una imagen y la muestra por pantalla
china = load_sample_image("china.jpg")
plt.imshow(china)
plt.grid(False)
plt.show()

#reduce el numero de pixeles de la imagen e indica el numero de colores en dicha imagen
image = china[::3, ::3]
n_colors = 2

#normaliza los valores de los pixeles
X = (image / 255.0).reshape(-1, 3)

#aplica k-means para agrupar los colores de la imagen
model = KMeans(n_colors)
labels = model.fit_predict(X)
colors = model.cluster_centers_

#vuelve a hacer la imagen
new_image = colors[labels].reshape(image.shape)
new_image = (255 * new_image).astype(np.uint8)

#muestra la imagen original y las nuevas
with sns.axes_style('white'):
    plt.figure()
    plt.imshow(image)
    plt.title('input')
    plt.figure()
    plt.imshow(new_image)
    plt.title('{0} colors'.format(n_colors))
    plt.show()
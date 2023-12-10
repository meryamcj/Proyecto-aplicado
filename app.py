import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from statsmodels.stats.diagnostic import normal_ad
from sklearn.decomposition import PCA
from openpyxl import Workbook
from scipy.stats import f_oneway
import imageio
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_predict
from scipy.stats import norm, kurtosis
import matplotlib.patheffects as path_effects
import matplotlib.font_manager
from matplotlib import pyplot
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import imageio
import scipy
import warnings
from wordcloud import WordCloud
from datetime import datetime

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
rcParams['figure.figsize'] = 8,4
rcParams['axes.linewidth']=3
plt.rcParams['font.size'] = 8
plt.rcParams['mathtext.fontset'] = 'stixsans'
rcParams['font.family'] = 'sans-serif'
colors = ["#1b60a7","#dda827","#f5c48e","#fbfaea","#279333","#83c0e9","#f6e2b9"]
sns.set_palette(sns.color_palette(colors))

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

df = pd.read_csv('data.csv')
# Streamlit app
st.title('App Proyecto Aplicado')

# Display the sample DataFrame
st.subheader(' DataFrame:')
st.dataframe(df)

# Create a scatter plot using Matplotlib
st.subheader('PCA:')
data = df

from sklearn.decomposition import PCA

df = data[["pundep","asignaturas"]]
com_names = ["pundep","asignaturas"]
df = df.dropna()
print(df.shape[0])

# You must normalize the data before applying the fit method
df =(df - df.mean()) / df.std()

pca = PCA(n_components=df.shape[1])
pca.fit(df)

# Reformat and view results
loadings = pd.DataFrame(pca.components_.T, columns = ['PC%s' % _ for _ in range(len(df.columns))], index = df.columns)
#print(loadings)

# Display the loadings DataFrame
st.subheader('Principal Component Loadings:')
st.dataframe(loadings)

# Bar plot of explained variance ratio
fig, ax = plt.subplots()
ax.bar(np.arange(0, len(loadings.columns), 1), pca.explained_variance_ratio_)
ax.set_xticks(np.arange(0, len(loadings.columns), 1))
ax.set_xticklabels(loadings.columns, rotation=90, fontsize=12)
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
ax.set_ylabel('Varianza explicada', fontsize=16)
ax.set_xlabel('Principal Components', fontsize=16)

# Display the bar plot
st.subheader('Explained Variance Ratio:')
st.pyplot(fig)

# Display the cumulative explained variance
st.subheader('Total Explained Variance:')
st.write(f'Total explained variance: {sum(pca.explained_variance_ratio_)}')


# elbow and kmeans

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Sample data creation for demonstration purposes
#data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
X = pd.DataFrame(data, columns=['pundep', 'asignaturas'])

st.title('K-Means Clustering')

# Choose a range of cluster numbers to test
cluster_range = range(1, 11)
inertia_values = []

# Calculate inertia for each cluster number
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve using Streamlit's line_chart
st.subheader('Elbow Method for Optimal k')
st.line_chart(zip(cluster_range, inertia_values))

# Perform KMeans clustering with chosen k
k = st.slider('Choose the number of clusters (k):', min_value=1, max_value=10, value=5)
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# Plot decision boundaries using Matplotlib
st.subheader(f'K-Means Clustering with {k} clusters')
fig, ax = plt.subplots()
scatter = ax.scatter(X['pundep'], X['asignaturas'], c=y_pred, cmap='viridis')
ax.set_xlabel('pundep')
ax.set_ylabel('asignaturas')
ax.set_title(f'K-Means Clustering with {k} clusters')
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)

# Display the Matplotlib plot using st.pyplot
st.pyplot(fig)

#Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Sample DataFrame creation for demonstration purposes
#data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
df = pd.DataFrame(data, columns=['pundep', 'asignaturas'])

st.title('Hierarchical Clustering')

# Choose the number of clusters (k) for hierarchical clustering
k = st.slider('Choose the number of clusters (k):', min_value=2, max_value=10, value=5)

# Perform hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=k)
labels = agg_clustering.fit_predict(df)

# Visualize the dendrogram
st.subheader('Hierarchical Clustering Dendrogram')
fig, ax = plt.subplots()
linkage_matrix = linkage(df, method='ward')
dendrogram(linkage_matrix, ax=ax)
ax.set_xlabel('pundep')
ax.set_ylabel('asignaturas')
st.pyplot(fig)

# Visualize the clusters
st.subheader('Hierarchical Clustering')
fig, ax = plt.subplots()
scatter = ax.scatter(df['pundep'], df['asignaturas'], c=labels, cmap='viridis')
ax.set_xlabel('pundep')
ax.set_ylabel('asignaturas')
ax.set_title('Hierarchical Clustering')
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)
st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Sample DataFrame creation for demonstration purposes
#df = pd.DataFrame(data, columns=data.columns)
numeric_columns = data.select_dtypes(include=np.number).columns
df = pd.DataFrame(data[numeric_columns])

st.title('Hierarchical Clustering')

# Choose the variables for hierarchical clustering
selected_columns = st.multiselect('Choose variables for clustering:', df.columns)

# Ensure that at least 2 variables are selected
if len(selected_columns) < 2:
    st.warning('Please select at least 2 variables for clustering.')
else:
    # Create a DataFrame with selected columns
    selected_df = df[selected_columns]

    # Choose the number of clusters (k) for hierarchical clustering
    slid = st.slider('Choose the number of clusters (k):', min_value=2, max_value=11, value=6)

    # Perform hierarchical clustering
    agg_clustering = AgglomerativeClustering(n_clusters=slid)
    labels = agg_clustering.fit_predict(selected_df)

    # Visualize the dendrogram
    st.subheader('Hierarchical Clustering Dendrogram')
    fig, ax = plt.subplots()
    linkage_matrix = linkage(selected_df, method='ward')
    dendrogram(linkage_matrix, ax=ax)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cluster Distance')
    st.pyplot(fig)

    # Visualize the clusters
    st.subheader('Hierarchical Clustering')
    fig, ax = plt.subplots()
    scatter = ax.scatter(selected_df.iloc[:, 0], selected_df.iloc[:, 1], c=labels, cmap='viridis')
    ax.set_xlabel(selected_columns[0])
    ax.set_ylabel(selected_columns[1])
    ax.set_title('Hierarchical Clustering')
    legend = ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.add_artist(legend)
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

# Import dataset
df = pd.read_excel('dataset_processed.xlsx')

# Header Interface
st.header("isi dataset")
st.write(df)

# Clustering process
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

# Display the elbow plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

# Create the slider for selecting the number of clusters (K)
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster: ", 2, 10, 1, 1)

# Scatter plot function
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    sns.scatterplot(x=df['Amount'], y=df['Recency'], hue=X['Labels'], palette=sns.color_palette('hls', n_colors=n_clust))
    sns.scatterplot(x=df['Amount'], y=df['Frequency'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust))
    sns.scatterplot(x=df['Frequency'], y=df['Recency'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust))
    sns.scatterplot(x=df['Amount'] ,y=df['Frequency'], hue = X['Recency'])

    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)

k_means(clust)

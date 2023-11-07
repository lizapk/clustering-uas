import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_excel('dataset_processed.xlsx')
df = df.drop(['Unnamed: 0'],axis=1)

# Header Interface
st.header("Isi dataset")
st.write(df)

# Create the slider for selecting the number of clusters (K)
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster: ", 2, 10, 1, 1)

# Scatter plot function
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(df)
    df['Labels'] = kmean.labels_

    st.subheader('Cluster Plot')
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    sns.scatterplot(x=df['Amount'], y=df['Recency'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust), ax=axes[0])
    axes[0].set_title('Amount vs Recency')

    sns.scatterplot(x=df['Amount'], y=df['Frequency'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust), ax=axes[1])
    axes[1].set_title('Amount vs Frequency')

    sns.scatterplot(x=df['Amount'], y=df['Frequency'], hue=df['Recency'], ax=axes[2])
    axes[2].set_title('Amount vs Frequency vs Recency')

    st.pyplot(fig)

# Clustering process and scatter plot
if st.button("Proses Klasterisasi dan Tampilkan Plot"):
    clusters = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i).fit(df)
        clusters.append(km.inertia_)

    # Display the elbow plot
    st.subheader("Mencari Elbow")
    st.line_chart(clusters, use_container_width=True)

    k_means(clust)

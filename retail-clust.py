import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data
df = pd.read_excel('processed_scale.xlsx')
df = df.drop(['Unnamed: 0'], axis=1)

df2 = pd.read_excel('processed_nonscale.xlsx')
df2 = df2.drop(['Unnamed: 0'], axis=1)

# Header Interface
st.header("Isi dataset")
st.write(df2)

# Create the slider for selecting the number of clusters (K)
st.sidebar.subheader("Nilai jumlah K")
n_clust = st.sidebar.slider("Pilih jumlah cluster: ", 2, 4, 2, 1)

# Create a selection box for scatter plot type
selected_plot = st.selectbox("Pilih Visualisasi Klasterisasi Data:", 
                             ["Total Belanja dan Terakhir Kali Belanja", 
                              "Total Belanja dan Frekuensi Belanja", 
                              "Total Belanja, Frekuensi Belanja dan Terakhir Kali Belanja"])

# Scatter plot functions
def plot_amount_vs_recency(n_clust):
    kmean = KMeans(n_clusters=n_clust, n_init=10).fit(df)
    df['Labels'] = kmean.labels_

    st.subheader('Total Belanja dan Terakhir Kali Belanja')
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(x=df['Amount'], y=df['Recency'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust), ax=ax)
    st.pyplot(fig)

def plot_amount_vs_frequency(n_clust):
    kmean = KMeans(n_clusters=n_clust, n_init=10).fit(df)
    df['Labels'] = kmean.labels_

    st.subheader('Total Belanja dan Frekuensi Belanja')
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(x=df['Amount'], y=df['Frequency'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust), ax=ax)
    st.pyplot(fig)

def plot_amount_vs_frequency_vs_recency(n_clust):
    kmean = KMeans(n_clusters=n_clust, n_init=10).fit(df)
    df['Labels'] = kmean.labels_

    st.subheader('Total Belanja, Frekuensi Belanja dan Terakhir Kali Belanja')
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(x=df['Amount'], y=df['Frequency'], hue=df['Recency'], ax=ax)
    st.pyplot(fig)

# Clustering process and scatter plot
if st.button("Proses Klasterisasi dan Tampilkan Plot"):
    clusters = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, n_init=10).fit(df)
        clusters.append(km.inertia_)

    # Display the elbow plot
    st.subheader("Mencari Elbow")
    st.line_chart(clusters, use_container_width=True)

    if selected_plot == "Total Belanja dan Terakhir Kali Belanja":
        plot_amount_vs_recency(n_clust)
    elif selected_plot == "Total Belanja dan Frekuensi Belanja":
        plot_amount_vs_frequency(n_clust)
    elif selected_plot == "Total Belanja, Frekuensi Belanja dan Terakhir Kali Belanja":
        plot_amount_vs_frequency_vs_recency(n_clust)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px

# Load data
day_df = pd.read_csv("https://raw.githubusercontent.com/rajifPy/dashbord_bikesharing/main/day.csv")
hour_df = pd.read_csv("https://raw.githubusercontent.com/rajifPy/dashbord_bikesharing/main/hour.csv")

custom_palette = ['#d0deec', '#97b2cc', '#8a9296']
bulan = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_df['month'] = pd.Categorical(day_df['mnth'], categories=bulan, ordered=True)

# Mengelompokkan berdasarkan bulan dan tahun
monthly_counts = day_df.groupby(by=["mnth", "yr"], observed=False).agg({
    "cnt": "sum"
}).reset_index()

seasonal_usage = day_df.groupby('season', observed=False).sum(numeric_only=True).reset_index()

seasonal_usage_year = day_df.groupby(['yr', 'season'], observed=False).sum(numeric_only=True).reset_index()

colors = ['#8C1C04', '#b7d657', '#ffc800', '#00b4cb']

# Menyiapkan daily_rent_df
def create_daily_rent_df(df):
    daily_rent_df = df.groupby(by='dteday', observed=False).agg({
        'cnt': 'sum'
    }).reset_index()
    return daily_rent_df

# Menyiapkan daily_casual_rent_df
def create_daily_casual_rent_df(df):
    daily_casual_rent_df = df.groupby(by='dteday', observed=False).agg({
        'casual': 'sum'
    }).reset_index()
    return daily_casual_rent_df

# Menyiapkan daily_registered_rent_df
def create_daily_registered_rent_df(df):
    daily_registered_rent_df = df.groupby(by='dteday', observed=False).agg({
        'registered': 'sum'
    }).reset_index()
    return daily_registered_rent_df

# Menyiapkan season_rent_df
def create_season_rent_df(df):
    season_rent_df = df.groupby(by='season', observed=False)[['registered', 'casual']].sum().reset_index()
    return season_rent_df

# Menyiapkan monthly_rent_df
def create_monthly_rent_df(df):
    monthly_rent_df = df.groupby(by='month', observed=False).agg({
        'cnt': 'sum'
    })
    ordered_months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    monthly_rent_df = monthly_rent_df.reindex(ordered_months, fill_value=0)
    return monthly_rent_df

# Menyiapkan weekday_rent_df
def create_weekday_rent_df(df):
    weekday_rent_df = df.groupby(by='weekday', observed=False).agg({
        'cnt': 'sum'
    }).reset_index()
    return weekday_rent_df

# Menyiapkan workingday_rent_df
def create_workingday_rent_df(df):
    workingday_rent_df = df.groupby(by='workingday', observed=False).agg({
        'cnt': 'sum'
    }).reset_index()
    return workingday_rent_df

# Panggil fungsi untuk membuat data frame
daily_rent_df = create_daily_rent_df(day_df)
daily_casual_rent_df = create_daily_casual_rent_df(day_df)
daily_registered_rent_df = create_daily_registered_rent_df(day_df)
monthly_rent_df = create_monthly_rent_df(day_df)

# Membuat judul
st.header('🔥muhammad rajif al farikhi🔥')
st.subheader('[Bike Sharing]', divider='rainbow')

# Membuat jumlah penyewaan harian
st.subheader('Peminjam Seepeda')
col1, col2, col3 = st.columns(3)

with col1:
    daily_rent_casual = daily_casual_rent_df['casual'].sum()
    st.metric('Peminjam Biasa', value=daily_rent_casual)

with col2:
    daily_rent_registered = daily_registered_rent_df['registered'].sum()
    st.metric('Peminjam Terdaftar', value=daily_rent_registered)

with col3:
    daily_rent_total = daily_rent_df['cnt'].sum()
    st.metric('Total Peminjam', value=daily_rent_total)
    
st.divider()

selected_chart = st.selectbox('Pilih Grafik',
                            ('Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca',
                                                         'Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca',
                            'Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun',
                            'Jumlah Penyewaan Sepeda berdasarkan Musim',
                            'Analisis Clustering'))

if selected_chart == 'Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca':
    st.subheader('Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca')

    fig = px.bar(day_df, x='weathersit', y='cnt', color='weathersit', color_discrete_sequence=custom_palette)
    fig.update_layout(title='Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca', xaxis_title='Kondisi Cuaca', yaxis_title='Jumlah Pengguna Sepeda')
    st.plotly_chart(fig)

elif selected_chart == 'Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun':
    st.subheader('Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun')

    fig = px.bar(monthly_counts, x="mnth", y="cnt", color="yr", barmode="group")
    fig.update_layout(title='Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun', xaxis_title='Bulan', yaxis_title='Jumlah', legend_title="Tahun")
    st.plotly_chart(fig)

    fig = px.line(monthly_counts, x="mnth", y="cnt", color="yr", markers=True)
    fig.update_layout(title='Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun', xaxis_title='Bulan', yaxis_title='Jumlah', legend_title="Tahun")
    st.plotly_chart(fig)

elif selected_chart == 'Jumlah Penyewaan Sepeda berdasarkan Musim':
    st.subheader('Jumlah Penyewaan Sepeda berdasarkan Musim')

    fig = px.bar(seasonal_usage, x='season', y='cnt', color='season', color_discrete_sequence=colors)
    fig.update_layout(title='Jumlah Penyewaan Sepeda berdasarkan Musim', xaxis_title='Musim', yaxis_title='Jumlah', showlegend=False)
    st.plotly_chart(fig)

    fig = px.bar(seasonal_usage_year, x='season', y='cnt', color='yr')
    fig.update_layout(title='Jumlah Peminjaman Sepeda berdasarkan Musim untuk Setiap Tahun', xaxis_title='Musim', yaxis_title='Jumlah', legend_title='Tahun')
    st.plotly_chart(fig)

elif selected_chart == 'Analisis Clustering':
    st.subheader('Analisis Clustering')

    # Memilih fitur untuk clustering
    features = ['temp', 'hum']
    X_cluster = day_df[features]

    # Menentukan jumlah cluster
    num_clusters = st.slider("Jumlah Cluster:", min_value=2, max_value=10, value=3, step=1)

    # Melakukan clustering dengan K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X_cluster)
    cluster_labels = kmeans.predict(X_cluster)

    # Menambahkan hasil clustering ke dalam dataframe
    day_df['cluster'] = cluster_labels

    # Memvisualisasikan hasil clustering
    fig = px.scatter(day_df, x='temp', y='hum', color='cluster', 
                     title='Clustering of Temperature vs Humidity', 
                     labels={'temp': 'Temperature', 'hum': 'Humidity', 'cluster': 'Cluster'})
    st.plotly_chart(fig)


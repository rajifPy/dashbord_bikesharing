import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Load data
day_df = pd.read_csv("C:/Users/muham/OneDrive - Universitas Airlangga/ML - Bangkit/DiCoding/Analisis Data Python/day.csv")
hour_df = pd.read_csv("C:/Users/muham/OneDrive - Universitas Airlangga/ML - Bangkit/DiCoding/Analisis Data Python/hour.csv")

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

# # Membuat jumlah penyewaan bulanan
# st.subheader('Peminjam Bulanan Berdasarkan Rentang Waktu')

# # Plot monthly bike rentals
# fig, ax = plt.subplots(figsize=(30, 10))
# ax.plot(
#     monthly_rent_df.index,
#     monthly_rent_df['cnt'],
#     marker='o', 
#     linewidth=2,
#     color='tab:blue'
# )

# for index, row in enumerate(monthly_rent_df['cnt']):
#     ax.text(index, row + 1, str(row), ha='center', va='bottom', fontsize=12)

# ax.tick_params(axis='x', labelsize=25, rotation=45)
# ax.tick_params(axis='y', labelsize=20)
# plt.xlabel('Bulan')
# plt.ylabel('Jumlah Pengguna Sepeda')

# # Show plot in Streamlit
# st.pyplot(fig)

selected_chart = st.selectbox('Pilih Grafik',
                            ('Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca',
                            'Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun',
                            'Jumlah Penyewaan Sepeda berdasarkan Musim',
                            'Analisis Clustering'))

if selected_chart == 'Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca':
    st.subheader('Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=day_df,
        x='weathersit',
        y='cnt',
        palette=custom_palette,
        ax=ax
    )
    plt.title('Jumlah Pengguna Sepeda berdasarkan Kondisi Cuaca')
    plt.xlabel('Kondisi Cuaca')
    plt.ylabel('Jumlah Pengguna Sepeda')
    st.pyplot(fig)

elif selected_chart == 'Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun':
    st.subheader('Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=monthly_counts,
        x="mnth",
        y="cnt",
        hue="yr",
        palette="tab10",
        ax=ax
    )
    plt.title("Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun")
    plt.xlabel("Bulan")
    plt.ylabel("Jumlah")
    plt.legend(title="Tahun", loc="upper right")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=monthly_counts,
        x="mnth",
        y="cnt",
        hue="yr",
        palette="tab10",
        marker="o",
        ax=ax
    )
    plt.title("Jumlah Pengguna Sepeda per Bulan untuk Setiap Tahun")
    plt.xlabel("Bulan")
    plt.ylabel("Jumlah")
    plt.legend(title="Tahun", loc="upper right")
    st.pyplot(fig)

elif selected_chart == 'Jumlah Penyewaan Sepeda berdasarkan Musim':
    st.subheader('Jumlah Penyewaan Sepeda berdasarkan Musim')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=seasonal_usage,
        x='season',
        y='cnt',
        hue='season',
        palette=colors,
        ax=ax
    )
    plt.xlabel("Musim")
    plt.ylabel("Jumlah")
    plt.title("Jumlah Penyewaan Sepeda berdasarkan Musim")
    plt.legend([],[], frameon=False)  # Hapus legenda kosong
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=seasonal_usage_year,
        x='season',
        y='cnt',
        hue='yr',
        ax=ax
    )
    plt.xlabel("Musim")
    plt.ylabel("Jumlah")
    plt.title("Jumlah Peminjaman Sepeda berdasarkan Musim untuk Setiap Tahun")
    plt.legend(title='Tahun')
    st.pyplot(fig)

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

    # Memvisualisasikan hasil clustering
    plt.figure(figsize=(10, 6))
    plt.scatter(X_cluster['temp'], X_cluster['hum'], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')
    plt.title('Clustering of Temperature vs Humidity')
    st.pyplot(plt)

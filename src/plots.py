import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_gender_distribution(data):
    """Mostra la comparativa de puntuacions per gènere."""
    rating_counts = data['gender'].value_counts()
    
    plt.figure(figsize=(12, 5))
    
    # Gràfic de barres
    plt.subplot(1, 2, 1)
    plt.bar(rating_counts.index, rating_counts.values, color=['cornflowerblue', 'salmon'])
    plt.title('Total Ratings by Gender')
    
    # Boxplot
    plt.subplot(1, 2, 2)
    ratings_m = data[data['gender'] == 'M']['rating']
    ratings_f = data[data['gender'] == 'F']['rating']
    plt.boxplot([ratings_m, ratings_f], tick_labels=['Male', 'Female'], patch_artist=True)
    plt.title('Rating Distribution')
    plt.show()

def plot_genre_preferences(data):
    """Mostra els gèneres preferits per homes i dones."""
    data_copy = data.copy()
    data_copy['genre'] = data_copy['genres'].str.split('|')
    df_exploded = data_copy.explode('genre')
    
    # Top 10 per gènere
    plt.figure(figsize=(15, 6))
    for i, (g, col) in enumerate([('M', 'cornflowerblue'), ('F', 'salmon')], 1):
        plt.subplot(1, 2, i)
        top_genres = df_exploded[df_exploded['gender'] == g]['genre'].value_counts().head(10)
        top_genres.sort_values().plot(kind='barh', color=col)
        plt.title(f'Top 10 Genres ({ "Males" if g=="M" else "Females" })')
    plt.tight_layout()
    plt.show()

def plot_gender_gap(data, min_ratings=100):
    """Mostra les pel·lícules amb més diferència de gust entre gèneres."""
    pivot = data.pivot_table(index='title', columns='gender', values='rating', aggfunc='mean')
    counts = data.groupby('title')['rating'].count()
    pivot = pivot.join(counts.rename('total')).query(f'total >= {min_ratings}').dropna()
    
    pivot['diff'] = pivot['M'] - pivot['F']
    
    plt.figure(figsize=(12, 6))
    top_m = pivot.sort_values('diff', ascending=False).head(10)
    top_f = pivot.sort_values('diff', ascending=True).head(10)
    
    plt.subplot(1, 2, 1)
    top_m['diff'].sort_values().plot(kind='barh', color='cornflowerblue')
    plt.title('Preferred by Men (M > F)')
    
    plt.subplot(1, 2, 2)
    top_f['diff'].abs().sort_values().plot(kind='barh', color='salmon')
    plt.title('Preferred by Women (F > M)')
    plt.tight_layout()
    plt.show()
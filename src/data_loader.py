import pandas as pd

import pandas as pd

def load_data(path='ml-1m/') -> pd.DataFrame:
    """Carrega i fusiona users, ratings i movies."""
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(f'{path}users.dat', sep='::', header=None, names=unames, engine='python')
    
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(f'{path}ratings.dat', sep='::', header=None, names=rnames, engine='python')
    
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(f'{path}movies.dat', sep='::', header=None, names=mnames, engine='python', encoding='latin-1')
    
    return pd.merge(pd.merge(ratings, users), movies)

def build_counts_table(df):
    """Crea la matriu User-Item."""
    return df.pivot_table('rating', index='user_id', columns='movie_id')

def get_count(df, user_id, movie_id) -> float:
    return df.loc[user_id, movie_id]
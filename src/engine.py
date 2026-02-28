import pandas as pd
import numpy as np

def predict_ratings(df_counts, user_id, sim_mx, m_neighbors):
    """Càlcul de mitjana ponderada (Weighted Average)."""
    user_sims = sim_mx.loc[user_id].drop(user_id)
    top_m = user_sims.sort_values(ascending=False).head(m_neighbors)
    
    if top_m.sum() <= 0: return {}
    
    weights = top_m / top_m.sum()
    neighbor_ratings = df_counts.loc[weights.index]
    
    numerator = neighbor_ratings.mul(weights, axis='index').sum(axis=0)
    denominator = neighbor_ratings.notna().mul(weights, axis='index').sum(axis=0)
    
    predictions = (numerator / denominator).replace([np.inf, -np.inf], np.nan).dropna()
    
    seen = df_counts.loc[user_id].dropna().index
    return predictions.drop(seen, errors='ignore').sort_values(ascending=False).to_dict()

def get_recommendations(df_all, user_id, sim_mx, n_recs, m_neighbors):
    from .data_loader import build_counts_table
    df_counts = build_counts_table(df_all)
    preds = predict_ratings(df_counts, user_id, sim_mx, m_neighbors)
    
    return pd.DataFrame.from_dict(preds, orient='index', columns=['score']) \
             .reset_index().rename(columns={'index': 'movie_id'}) \
             .head(n_recs)
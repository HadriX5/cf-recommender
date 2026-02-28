import numpy as np
import pandas as pd

def evaluate_mae(train_df, test_df, sim_mx, m_neighbors):
    """Calcula el Mean Absolute Error."""
    from .engine import predict_ratings
    from .data_loader import build_counts_table
    
    counts_train = build_counts_table(train_df)
    errors = []
    
    for user_id in test_df['user_id'].unique():
        preds = predict_ratings(counts_train, user_id, sim_mx, m_neighbors)
        actuals = test_df[test_df['user_id'] == user_id]
        
        for _, row in actuals.iterrows():
            if row['movie_id'] in preds:
                errors.append(abs(preds[row['movie_id']] - row['rating']))
                
    return np.mean(errors) if errors else 0.0
import numpy as np

def dist_euclid(x, y):
    return np.linalg.norm(x - y)

def sim_euclid(vec1, vec2, n_total_items):
    """Càlcul de similitud normalitzada per la mida del dataset."""
    dist = dist_euclid(vec1, vec2)
    return (1. / (1. + dist)) * (len(vec1) / n_total_items)

def similarity_matrix_vectorized(df_counts):
    """Versió optimitzada sense bucles (Similarity Matrix 2)."""
    df = df_counts.fillna(0).values
    mask = (df != 0).astype(float)
    
    df2 = df ** 2
    creuat = df @ df.T
    masked_df = df2 @ mask.T
    
    mat_dist = np.maximum(masked_df - 2 * creuat + masked_df.T, 0)
    mat_dist = np.sqrt(mat_dist)
    
    mat_sim = 1 / (1 + mat_dist)
    common_n = mask @ mask.T
    pond = common_n / df.shape[1]
    
    sim = mat_sim * pond
    sim[common_n == 0] = 0
    np.fill_diagonal(sim, 0.0)
    return sim
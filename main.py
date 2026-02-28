import pandas as pd
from src.data_loader import load_data, build_counts_table
from src.similarity import similarity_matrix_vectorized
from src.engine import get_recommendations

def main():
    print("--- 🎬 MovieLens Recommender System ---")
    data = load_data()
    counts = build_counts_table(data)
    
    print("Computing similarity matrix (vectorized)...")
    sim_array = similarity_matrix_vectorized(counts)
    sim_df = pd.DataFrame(sim_array, index=counts.index, columns=counts.index)
    
    USER_ID = 42
    print(f"\nGenerating top recommendations for User {USER_ID}...")
    recs = get_recommendations(data, USER_ID, sim_df, n_recs=10, m_neighbors=20)
    print(recs)

if __name__ == "__main__":
    main()
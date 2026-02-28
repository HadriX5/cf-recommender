# MovieLens Recommendation Engine

An end-to-end Collaborative Filtering system built with Python, Pandas, and NumPy. This project implements a user-item recommendation engine and analyzes rating behaviors across different demographics.

## 🚀 Key Features

- **Vectorized Similarity**: High-performance similarity matrix calculation using NumPy broadcasting (avoiding slow Python loops).
- **Collaborative Filtering**: Weighted average prediction model based on Euclidean distance similarity.
- **Data Visualization**: Detailed analysis of rating distributions and genre preferences by gender.
- **Evaluation Pipeline**: Implements Mean Absolute Error (MAE) calculation using a hold-out validation strategy.
- **Hypothesis Testing**: Advanced analysis to compare general models vs. gender-segregated models.

## 📊 Visualizations

The system includes a suite of analytical tools to explore:

- Total ratings and distributions by gender.
- Top-rated genres for Men vs. Women.
- "Gender Gap" in movies: Identify films with the highest rating discrepancy between genders.

## 🧠 Technical Highlights

### Vectorized Matrix Similarity

The core of the engine uses a highly optimized function to compute user similarities without explicit iteration:

$$
Dist(A, B) = \sqrt{\sum (A_i - B_i)^2}
$$

This is implemented via matrix multiplication ($A \cdot A^T$) to ensure scalability even with thousands of users.

## 📂 Project Structure

- `data_loader.py`: Handles the MovieLens-1M triple-merge (Users + Ratings + Movies).
- `similarity.py`: Contains the logic for Euclidean distance and similarity normalization.
- `engine.py`: The recommendation core that calculates weighted averages for predictions.
- `evaluation.py`: Tools for splitting data (80/20) and calculating MAE.

## 🛠️ Setup

1. Download the MovieLens-1M dataset and place it in the `ml-1m/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `python main.py`

# content_recommender/recommender.py
"""Core classes and logic for the hybrid recommendation engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field # Added field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

@dataclass
class Recommendation:
    """Represents a single recommended series with score and metadata."""
    series_id: int
    name: str
    score: float  # Combined score for hybrid model
    metadata: Dict[str, str]
    # Optional scores for analysis/debugging
    content_score: Optional[float] = field(default=None, compare=False)
    collab_score: Optional[float] = field(default=None, compare=False)

    def to_dict(self) -> Dict[str, object]:
        """Serializes the recommendation to a dictionary."""
        data = {
            "series_id": self.series_id,
            "name": self.name,
            "score": round(self.score, 4),
            **self.metadata,
        }
        # Add optional scores if they exist
        if self.content_score is not None: data["content_score"] = round(self.content_score, 4)
        if self.collab_score is not None: data["collab_score"] = round(self.collab_score, 4)
        return data

class ContentRecommender:
    """
    Hybrid recommendation engine combining content-based filtering (TF-IDF)
    and user-based collaborative filtering.
    """
    # --- Configuration: Adjust to match your CSV column names ---
    _ID_COL: str = "series_id"
    _NAME_COL: str = "name"
    _FEATURE_COLS: List[str] = ["genre", "narrative_format", "style"]
    # -----------------------------------------------------------
    _REQUIRED_PRODUCT_COLS: List[str] = [_ID_COL, _NAME_COL] + _FEATURE_COLS
    _REQUIRED_RATING_COLS: List[str] = ["user_id", "series_id", "rating"]

    def __init__(self, products_path: Path | str, ratings_path: Path | str) -> None:
        """Initializes the recommender by loading data and pre-calculating matrices."""
        self._load_and_validate_products(Path(products_path))
        self._build_content_model()
        self._load_and_build_collaborative_model(Path(ratings_path))

    def _load_and_validate_products(self, path: Path) -> None:
        """Loads and validates the products (series catalog) CSV."""
        if not path.exists(): raise FileNotFoundError(f"Products file not found: {path}")
        self._products: pd.DataFrame = pd.read_csv(path, encoding="utf-8")
        self._validate_dataframe(self._products, self._REQUIRED_PRODUCT_COLS, path.name)
        # Type casting and NaN handling
        self._products[self._ID_COL] = self._products[self._ID_COL].astype(int)
        self._products[self._NAME_COL] = self._products[self._NAME_COL].fillna('Unknown Title').astype(str)
        for col in self._FEATURE_COLS: self._products[col] = self._products[col].fillna('').astype(str)
        # Check for duplicate IDs after loading
        self._check_duplicate_ids(self._products, self._ID_COL, path.name)
        # Create ID <-> Name mappings
        self._id_to_name: Dict[int, str] = pd.Series(self._products[self._NAME_COL].values, index=self._products[self._ID_COL]).to_dict()
        self._name_to_id: Dict[str, int] = {v: k for k, v in self._id_to_name.items()}

    def _build_content_model(self) -> None:
        """Builds the TF-IDF vectorizer and feature matrix from product features."""
        print("Building content model (TF-IDF)...")
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self._feature_matrix = self._vectorizer.fit_transform(self._combine_features(self._products))
        print(f"  TF-IDF Matrix Shape: {self._feature_matrix.shape}")

    def _load_and_build_collaborative_model(self, path: Path) -> None:
        """Loads user ratings, handles duplicates, builds utility matrix, and calculates user similarity."""
        if not path.exists(): raise FileNotFoundError(f"Ratings file not found: {path}")
        print(f"Loading user ratings from: {path.name}")
        df_ratings: pd.DataFrame = pd.read_csv(path)
        self._validate_dataframe(df_ratings, self._REQUIRED_RATING_COLS, path.name)

        # Filter ratings for products that exist in the catalog
        valid_series_ids = set(self._products[self._ID_COL])
        original_rating_count = len(df_ratings)
        df_ratings = df_ratings[df_ratings['series_id'].isin(valid_series_ids)]
        if len(df_ratings) < original_rating_count:
            print(f"  Warning: Filtered out {original_rating_count - len(df_ratings)} ratings for unknown series_ids.")

        # --- NOVA LINHA: TRATAR DUPLICATAS ---
        # If a user rated the same series multiple times, keep the average rating.
        print("Handling duplicate user-series ratings by averaging...")
        df_ratings = df_ratings.groupby(['user_id', 'series_id'])['rating'].mean().reset_index()
        # -------------------------------------

        print("Building utility matrix...")
        # Now pivot should work because duplicates are removed
        self._utility_matrix: pd.DataFrame = df_ratings.pivot(
            index='user_id',
            columns='series_id',
            values='rating'
        )
        print(f"  Utility Matrix Shape (raw): {self._utility_matrix.shape}")

        # Fill NaNs for similarity calculation
        self._utility_matrix_filled: pd.DataFrame = self._utility_matrix.fillna(0)

        print("Calculating user similarity matrix (cosine)...")
        user_similarity_matrix = 1 - pairwise_distances(self._utility_matrix_filled.values, metric='cosine')
        self._user_similarity_df = pd.DataFrame(user_similarity_matrix,
                                               index=self._utility_matrix_filled.index,
                                               columns=self._utility_matrix_filled.index)
        print("  User similarity matrix calculated.")

    def _validate_dataframe(self, df: pd.DataFrame, required_cols: List[str], filename: str) -> None:
        """Generic validation for required columns in a DataFrame."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"File '{filename}' is missing required columns: {', '.join(missing)}")

    def _check_duplicate_ids(self, df: pd.DataFrame, id_col: str, filename: str) -> None:
        """Checks for duplicate IDs in a DataFrame."""
        duplicates = df[id_col].duplicated()
        if duplicates.any():
            duplicate_ids = df.loc[duplicates, id_col].unique().tolist()
            raise ValueError(f"File '{filename}' contains duplicate IDs in column '{id_col}': {duplicate_ids}")

    @staticmethod
    def _normalize_token(value: str) -> str:
        """Cleans feature tokens."""
        cleaned = str(value).lower().replace("-", " ").replace("_", " ")
        return " ".join(cleaned.split())

    def _combine_features(self, products: pd.DataFrame) -> Iterable[str]:
        """Combines relevant feature columns into a single text string per product."""
        cols_to_combine = [self._NAME_COL] + self._FEATURE_COLS
        # Apply normalization to each specified column's value for each row
        return products.apply(
            lambda row: " ".join(self._normalize_token(str(row[col])) for col in cols_to_combine),
            axis=1
        )

    def build_profile_from_ratings(self, avaliacoes: Dict[str, int]) -> Tuple[np.ndarray, pd.Series]:
        """
        Creates the content profile vector and the user's rating Series from explicit ratings.
        Returns tuple: (content_profile_vector, user_rating_series)
        """
        if not isinstance(avaliacoes, dict):
            raise TypeError("Input 'avaliacoes' must be a dictionary.")

        # Map evaluated series names to their internal indices and IDs
        valid_evaluations: Dict[str, int] = {}
        indices_avaliados: List[int] = []
        rating_data: Dict[int, float] = {} # series_id -> rating

        for name, rating in avaliacoes.items():
            if name in self._name_to_id:
                series_id = self._name_to_id[name]
                # Find the DataFrame index corresponding to the series_id
                product_index = self._products[self._products[self._ID_COL] == series_id].index
                if not product_index.empty:
                    indices_avaliados.append(product_index[0])
                    rating_data[series_id] = float(rating)
                    valid_evaluations[name] = rating # Keep valid ones
            else:
                print(f"Warning: Evaluated series '{name}' not found in product catalog.")

        # --- Create User Rating Series (aligned with utility matrix columns) ---
        user_rating_series = pd.Series(index=self._utility_matrix.columns, dtype=float)
        for series_id, rating in rating_data.items():
            if series_id in user_rating_series.index:
                user_rating_series[series_id] = rating

        # --- Create Content Profile Vector ---
        num_valid_ratings = len(indices_avaliados)
        if num_valid_ratings == 0:
            print("Warning: No valid ratings provided or found in catalog.")
            return np.zeros((1, self._feature_matrix.shape[1])), user_rating_series

        vetores_avaliados = self._feature_matrix[indices_avaliados]
        notas_array = np.array([valid_evaluations[name] for name in self._products.loc[indices_avaliados, self._NAME_COL]]).reshape(-1, 1)

        # Calculate weights based on deviation from the mean rating
        if num_valid_ratings > 1:
            pesos = notas_array - np.mean(notas_array)
        else:
            pesos = np.array([[1.0]]) # Neutral weight for single rating

        soma_abs_pesos = np.sum(abs(pesos))
        # Use simple average if weights sum to near zero
        if soma_abs_pesos < 1e-9:
            print("Warning: Rating weights sum to zero. Using simple average for content profile.")
            content_profile_vector_matrix = np.mean(vetores_avaliados.toarray(), axis=0)
        else:
            # Weighted average using sparse matrix multiplication for efficiency
            content_profile_vector_matrix = np.sum(vetores_avaliados.multiply(pesos), axis=0) / soma_abs_pesos

        # Ensure correct shape (1, num_features)
        content_profile_vector = np.asarray(content_profile_vector_matrix).flatten().reshape(1, -1)

        return content_profile_vector, user_rating_series

    def _predict_collaborative_scores(self, user_rating_series: pd.Series, k_neighbors: int) -> pd.Series:
        """Predicts ratings for unrated items using user-based collaborative filtering."""
        
        # Calculate similarity between the new user and all users in the matrix
        # Fill NaN in the new user's ratings with 0 for similarity calculation
        user_ratings_filled = user_rating_series.fillna(0).values.reshape(1, -1)
        
        similarities = 1 - pairwise_distances(user_ratings_filled, self._utility_matrix_filled.values, metric='cosine').flatten()
        
        # Get top K similar users (neighbors)
        sim_scores = pd.Series(similarities, index=self._user_similarity_df.index)
        # Exclude self-similarity if the user somehow existed in the matrix (safety check)
        # sim_scores = sim_scores.drop(user_id, errors='ignore')
        sim_scores = sim_scores[sim_scores > 0] # Consider only positively correlated users
        top_k_neighbors = sim_scores.nlargest(k_neighbors)

        if top_k_neighbors.empty:
            print("Warning: No similar users found for collaborative filtering.")
            # Return empty Series with correct index
            return pd.Series(index=self._utility_matrix.columns[user_rating_series.isna()], dtype=float)

        # Get ratings of neighbors (from the original matrix with NaNs)
        neighbor_ratings = self._utility_matrix.loc[top_k_neighbors.index]

        # Calculate weighted average prediction for items the new user hasn't rated
        predicted_scores = {}
        # Iterate only over items the new user *hasn't* rated
        items_to_predict = self._utility_matrix.columns[user_rating_series.isna()]

        for item_id in items_to_predict:
            # Ratings given by neighbors to this specific item
            item_ratings_by_neighbors = neighbor_ratings[item_id].dropna() # Drop neighbors who didn't rate it
            if item_ratings_by_neighbors.empty:
                continue # No neighbors rated this item

            # Weights (similarities) of the neighbors who rated this item
            neighbor_weights = top_k_neighbors[item_ratings_by_neighbors.index]

            # Calculate weighted average
            weighted_sum = np.dot(item_ratings_by_neighbors.values, neighbor_weights.values)
            sum_of_weights = neighbor_weights.sum()

            if sum_of_weights > 1e-9:
                predicted_scores[item_id] = weighted_sum / sum_of_weights

        return pd.Series(predicted_scores, dtype=float)

    def recommend(
        self,
        profile_content_vector: np.ndarray,
        new_user_ratings: pd.Series, # User's rating vector (Series with NaNs)
        series_ja_avaliadas: List[str], # Names of series already rated
        top_n: int = 5,
        content_weight: float = 0.6, # Default weight for content score
        k_neighbors_collab: int = 15,
        return_dataframe: bool = False,
    ) -> List[Recommendation] | pd.DataFrame:
        """Generates hybrid recommendations combining content and collaborative scores."""

        # --- 1. Content Score Calculation ---
        if profile_content_vector.shape[1] != self._feature_matrix.shape[1]:
             raise ValueError("Content profile dimension mismatch.")
        content_scores_array = cosine_similarity(profile_content_vector, self._feature_matrix).flatten()
        # Use series NAME as index for easy merging later
        content_scores = pd.Series(content_scores_array, index=self._products[self._NAME_COL])
        # Normalize content scores to [0, 1] range
        min_c, max_c = content_scores.min(), content_scores.max()
        content_scores_norm = (content_scores - min_c) / (max_c - min_c) if (max_c - min_c) > 1e-9 else pd.Series(0.0, index=content_scores.index)

        # --- 2. Collaborative Score Calculation ---
        collab_predictions_by_id = self._predict_collaborative_scores(new_user_ratings, k_neighbors=k_neighbors_collab)
        # Map predicted series IDs to names
        collab_scores = collab_predictions_by_id.rename(index=self._id_to_name).dropna()
        # Normalize collaborative scores to [0, 1] range
        min_col, max_col = collab_scores.min(), collab_scores.max()
        collab_scores_norm = (collab_scores - min_col) / (max_col - min_col) if (max_col - min_col) > 1e-9 else pd.Series(0.0, index=collab_scores.index)

        # --- 3. Combine Scores ---
        collab_weight = 1.0 - content_weight
        # Create DataFrame from the normalized scores Series, aligning by index (series name)
        combined_scores_df = pd.DataFrame({
            'content_score': content_scores_norm,
            'collab_score': collab_scores_norm
        })
        # Fill missing scores (e.g., if collab filtering couldn't predict for an item) with 0
        combined_scores_df.fillna(0, inplace=True)
        # Calculate final weighted score
        combined_scores_df['final_score'] = (content_weight * combined_scores_df['content_score']) + \
                                            (collab_weight * combined_scores_df['collab_score'])

        # --- 4. Generate Final Ranking ---
        # Remove already rated series
        final_ranking = combined_scores_df[~combined_scores_df.index.isin(series_ja_avaliadas)]
        # Sort by the combined score
        final_ranking = final_ranking.sort_values('final_score', ascending=False)
        # Get top N results
        top_n_recommendations = final_ranking.head(top_n)

        # --- 5. Format Output ---
        if return_dataframe:
            # Return names and scores
            return top_n_recommendations[['final_score', 'content_score', 'collab_score']]

        # Format as list of Recommendation objects, merging with product metadata
        recommendations: List[Recommendation] = []
        recs_details = self._products.set_index(self._NAME_COL).loc[top_n_recommendations.index]
        recs_with_scores = recs_details.join(top_n_recommendations)
        recs_with_scores = recs_with_scores.reset_index()

        recs_with_scores.rename(columns={'index': self._NAME_COL}, inplace=True)

        for row in recs_with_scores.itertuples(index=False):
            metadata = {col: str(getattr(row, col, 'N/A')) for col in self._FEATURE_COLS}
            recommendations.append(Recommendation(
                 series_id=int(getattr(row, self._ID_COL)),
                 name=str(getattr(row, self._NAME_COL)),
                 score=float(getattr(row, 'final_score')), # Corrigido para final_score
                 metadata=metadata,
                 content_score=float(getattr(row, 'content_score')),
                 collab_score=float(getattr(row, 'collab_score'))
            ))
        return recommendations

    def get_available_series_names(self) -> List[str]:
        """Returns a sorted list of unique series names from the catalog."""
        if self._NAME_COL not in self._products.columns:
             print(f"Warning: Name column '{self._NAME_COL}' not found in products.")
             return []
        return sorted(self._products[self._NAME_COL].unique().tolist())

    def to_json(self, recommendations: Iterable[Recommendation]) -> str:
        """Serializes recommendations to a JSON string."""
        return json.dumps([rec.to_dict() for rec in recommendations], ensure_ascii=False, indent=2)
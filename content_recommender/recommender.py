from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Recommendation:
    """Representation of a product recommendation."""

    product_id: int
    name: str
    score: float
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "score": round(float(self.score), 4),
            **self.metadata,
        }


class ContentRecommender:
    """Content-based recommendation engine for marketplace products."""

    def __init__(self, products_path: Path | str) -> None:
        self.products_path = Path(products_path)
        if not self.products_path.exists():
            raise FileNotFoundError(f"Products file not found: {self.products_path}")

        self._products = pd.read_csv(self.products_path)
        self._vectorizer = TfidfVectorizer()
        self._feature_matrix = self._vectorizer.fit_transform(self._combine_features(self._products))

    @staticmethod
    def _combine_features(products: pd.DataFrame) -> Iterable[str]:
        return (
            (
                f"{row.category} {row.flavor} {row.benefit} "
                f"{row.name.replace('-', ' ')}"
            ).lower()
            for row in products.itertuples()
        )

    @property
    def products(self) -> pd.DataFrame:
        return self._products.copy()

    def available_options(self) -> Dict[str, List[str]]:
        """Return dictionaries of possible values for each feature."""

        return {
            "category": sorted(self._products["category"].unique().tolist()),
            "flavor": sorted(self._products["flavor"].unique().tolist()),
            "benefit": sorted(self._products["benefit"].unique().tolist()),
        }

    def build_profile(self, preferences: Dict[str, str]) -> str:
        """Create a textual representation of the user profile."""

        options = self.available_options()
        profile_tokens = []
        for key, value in preferences.items():
            if value is None:
                continue
            if key in options and value not in options[key]:
                raise ValueError(f"'{value}' is not a valid option for '{key}'.")
            profile_tokens.append(str(value))
        return " ".join(token.lower() for token in profile_tokens if token)

    def recommend(
        self,
        profile_text: str,
        top_n: int = 5,
        return_dataframe: bool = False,
    ) -> List[Recommendation] | pd.DataFrame:
        """Return the top-N product recommendations for the given profile."""

        if not profile_text.strip():
            raise ValueError("Profile text must not be empty.")

        profile_vector = self._vectorizer.transform([profile_text])
        similarity_scores = cosine_similarity(profile_vector, self._feature_matrix).flatten()
        ranking = self._products.assign(score=similarity_scores).sort_values(
            "score", ascending=False
        )
        ranking = ranking.head(top_n)

        if return_dataframe:
            return ranking

        recommendations = [
            Recommendation(
                product_id=int(row.product_id),
                name=str(row.name),
                score=float(row.score),
                metadata={
                    "category": str(row.category),
                    "flavor": str(row.flavor),
                    "benefit": str(row.benefit),
                },
            )
            for row in ranking.itertuples(index=False)
        ]
        return recommendations

    def to_json(self, recommendations: Iterable[Recommendation]) -> str:
        """Serialize recommendations to JSON."""

        return json.dumps([rec.to_dict() for rec in recommendations], ensure_ascii=False, indent=2)


def load_default_recommender() -> ContentRecommender:
    data_dir = Path(__file__).resolve().parent / "data"
    return ContentRecommender(data_dir / "products.csv")

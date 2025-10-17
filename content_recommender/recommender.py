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
    """Representation of a series recommendation."""

    series_id: int
    name: str
    score: float
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "series_id": self.series_id,
            "name": self.name,
            "score": round(float(self.score), 4),
            **self.metadata,
        }


class ContentRecommender:
    """Content-based recommendation engine for TV series."""

    _REQUIRED_COLUMNS = ("series_id", "name", "genre", "narrative_format", "style")

    def __init__(self, products_path: Path | str) -> None:
        self.products_path = Path(products_path)
        if not self.products_path.exists():
            raise FileNotFoundError(f"Products file not found: {self.products_path}")

        self._products = pd.read_csv(self.products_path, encoding="utf-8")
        self._validate_dataset()
        self._products = self._products[list(self._REQUIRED_COLUMNS)].copy()
        self._products["series_id"] = self._products["series_id"].astype(int)

        self._vectorizer = TfidfVectorizer()
        self._feature_matrix = self._vectorizer.fit_transform(
            self._combine_features(self._products)
        )

    def _validate_dataset(self) -> None:
        missing_columns = [
            column for column in self._REQUIRED_COLUMNS if column not in self._products.columns
        ]
        if missing_columns:
            raise ValueError(
                "Products dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
            
        if self._products[self._REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError("Products dataset contains empty values in required columns.")

        duplicated_ids = self._products["series_id"].duplicated()
        if duplicated_ids.any():
            duplicates = self._products.loc[duplicated_ids, "series_id"].tolist()
            raise ValueError(
                "Products dataset contains duplicated series_id values: "
                + ", ".join(map(str, duplicates))
            )

    @staticmethod
    def _normalize_token(value: str) -> str:
        cleaned = str(value).replace("-", " ").replace("_", " ")
        return " ".join(cleaned.split())

    @staticmethod
    def _combine_features(products: pd.DataFrame) -> Iterable[str]:
        return (
            (
                f"{ContentRecommender._normalize_token(row.genre)} "
                f"{ContentRecommender._normalize_token(row.narrative_format)} "
                f"{ContentRecommender._normalize_token(row.style)} "
                f"{ContentRecommender._normalize_token(row.name)}"
            ).lower()
            for row in products.itertuples()
        )

    @property
    def products(self) -> pd.DataFrame:
        return self._products.copy()

    def available_options(self) -> Dict[str, List[str]]:
        """Return dictionaries of possible values for each feature."""

        return {
            "genre": sorted(self._products["genre"].unique().tolist()),
            "narrative_format": sorted(
                self._products["narrative_format"].unique().tolist()
            ),
            "style": sorted(self._products["style"].unique().tolist()),
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
            profile_tokens.append(self._normalize_token(str(value)))
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
                series_id=int(row.series_id),
                name=str(row.name),
                score=float(row.score),
                metadata={
                    "genre": str(row.genre),
                    "narrative_format": str(row.narrative_format),
                    "style": str(row.style),
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


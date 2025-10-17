"""Interface de linha de comando para o recomendador de séries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from content_recommender import ContentRecommender


FIELD_LABELS = {
    "genre": "gênero",
    "narrative_format": "formato narrativo",
    "style": "estilo",
}


def _ask_option(label: str, options: List[str]) -> str:
    print(f"\nSelecione uma opção para {label} (ou pressione Enter para pular):")
    for idx, option in enumerate(options, start=1):
        print(f"  [{idx}] {option}")

    while True:
        choice = input(
            f"Número da opção desejada para {label}: "
        ).strip()
        if not choice:
            return ""
        if not choice.isdigit():
            print("Digite apenas o número da opção ou pressione Enter para pular.")
            continue
        idx = int(choice)
        if not 1 <= idx <= len(options):
            print("Opção inválida, tente novamente.")
            continue
        return options[idx - 1]


def _collect_preferences(recommender: ContentRecommender) -> Dict[str, str]:
    preferences: Dict[str, str] = {}
    options = recommender.available_options()
    for field, field_options in options.items():
        label = FIELD_LABELS.get(field, field)
        selected = _ask_option(label, field_options)
        if selected:
            preferences[field] = selected
    return preferences


def _print_recommendations(recommendations) -> None:
    print("\nSéries recomendadas:\n")
    for position, rec in enumerate(recommendations, start=1):
        print(f"{position}. {rec.name} (similaridade: {rec.score:.3f})")
        print(
            f"   Gênero: {rec.metadata['genre']} | "
            f"Formato: {rec.metadata['narrative_format']} | Estilo: {rec.metadata['style']}"
        )


def main() -> None:
    print("=" * 70)
    print("Guia de Séries – Sistema de Recomendação por Conteúdo")
    print("Descubra produções que combinam com o seu gosto")
    print("=" * 70)

    name = input("\nPara começar, informe seu nome: ").strip() or "Visitante"

    data_path = Path(__file__).resolve().parent / "content_recommender" / "data" / "products.csv"
    try:
        recommender = ContentRecommender(data_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nNão foi possível carregar o catálogo de séries: {error}")
        return

    print(
        f"\nOlá, {name}! Vamos montar seu perfil de preferências. "
        "Escolha as opções que mais combinam com seu gosto por séries."
    )

    profile_text = ""
    preferences: Dict[str, str] = {}
    while not profile_text:
        preferences = _collect_preferences(recommender)
        profile_text = recommender.build_profile(preferences)
        if not profile_text:
            print(
                "\nÉ necessário selecionar ao menos uma preferência para gerar recomendações."
            )

    recommendations = recommender.recommend(profile_text)

    _print_recommendations(recommendations)

    export = input("\nDeseja salvar as recomendações em JSON? (s/N): ").strip().lower()
    if export == "s":
        output_file = Path("recomendacoes.json")
        output_file.write_text(recommender.to_json(recommendations), encoding="utf-8")
        print(f"Recomendações salvas em {output_file.resolve()}")


if __name__ == "__main__":
    main()

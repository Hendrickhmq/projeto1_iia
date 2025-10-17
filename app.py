"""Interface de linha de comando para o recomendador de produtos naturais."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from content_recommender import ContentRecommender


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
        selected = _ask_option(field, field_options)
        if selected:
            preferences[field] = selected
    return preferences


def _print_recommendations(recommendations) -> None:
    print("\nProdutos recomendados:\n")
    for position, rec in enumerate(recommendations, start=1):
        print(f"{position}. {rec.name} (similaridade: {rec.score:.3f})")
        print(
            f"   Categoria: {rec.metadata['category']} | "
            f"Sabor: {rec.metadata['flavor']} | Benefício: {rec.metadata['benefit']}"
        )


def main() -> None:
    print("=" * 70)
    print("Bem-vindo ao Mercado Verde Viva")
    print("Sistema de recomendação por conteúdo para produtos naturais")
    print("=" * 70)

    name = input("\nPara começar, informe seu nome: ").strip() or "Visitante"

    data_path = Path(__file__).resolve().parent / "content_recommender" / "data" / "products.csv"
    recommender = ContentRecommender(data_path)

    print(
        f"\nOlá, {name}! Vamos montar seu perfil de preferências. "
        "Escolha as opções que mais combinam com você."
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

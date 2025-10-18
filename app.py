# app.py
"""Command-line interface for the hybrid series recommender."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

try:
    # Import directly from the recommender.py file in the package
    from content_recommender.recommender import ContentRecommender, Recommendation
except ImportError as e:
     print(f"Erro de Importação: {e}")
     print("Verifique se 'recommender.py' está dentro da pasta 'content_recommender'.")
     exit()
except Exception as e:
     print(f"Erro inesperado durante importação: {e}")
     exit()

# --- Configuration: Adjust file names if needed ---
NOME_ARQUIVO_PRODUTOS = "products.csv"
NOME_ARQUIVO_RATINGS = "user_ratings.csv"
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "content_recommender" / "data"
PRODUCTS_PATH = DATA_DIR / NOME_ARQUIVO_PRODUTOS
RATINGS_PATH = DATA_DIR / NOME_ARQUIVO_RATINGS

def _display_series_list(series_list: List[str]):
    """Displays the list of available series in formatted columns."""
    print("\n--- Catálogo de Séries Disponíveis ---")
    num_series = len(series_list)
    cols = 3 # Adjust columns based on terminal width if desired
    rows = (num_series + cols - 1) // cols
    for i in range(rows):
        line_items = []
        for j in range(cols):
            idx = i + j * rows
            if idx < num_series:
                line_items.append(f"[{idx+1:2d}] {series_list[idx]}")
        # Adjust spacing dynamically based on longest item in current line segment
        # Basic fixed spacing for simplicity here:
        print("   ".join(f"{item:<35}" for item in line_items))
    print("-" * 110) # Adjust separator width

def _collect_ratings_from_choice(recommender: ContentRecommender, num_to_choose: int = 5) -> Dict[str, int]:
    """Guides the user to select and rate a number of series."""
    print(f"\nPasso 1: Escolha {num_to_choose} séries da lista para avaliar (nota 1-5).")

    user_ratings: Dict[str, int] = {}
    available_series = recommender.get_available_series_names()

    if not available_series:
        print("Erro: Nenhuma série encontrada no catálogo carregado.")
        return {}

    _display_series_list(available_series)

    chosen_series_names: List[str] = []
    chosen_indices: set[int] = set()

    print(f"\nDigite o NÚMERO da série (1-{len(available_series)}). Digite 'fim' quando terminar a seleção.")
    while len(chosen_series_names) < num_to_choose:
        prompt = f"Escolha a série nº {len(chosen_series_names) + 1}/{num_to_choose} (ou 'fim'): "
        try:
            choice_str = input(prompt).strip()

            if choice_str.lower() == 'fim':
                 if not chosen_series_names: print("  >> Você precisa escolher pelo menos uma série."); continue
                 else: print("  >> Seleção finalizada."); break

            if not choice_str.isdigit(): print("  >> Entrada inválida. Digite o número."); continue
            choice_idx = int(choice_str) - 1

            if not (0 <= choice_idx < len(available_series)): print("  >> Número fora do intervalo."); continue
            if choice_idx in chosen_indices: print("  >> Série já escolhida."); continue

            selected_name = available_series[choice_idx]
            chosen_series_names.append(selected_name)
            chosen_indices.add(choice_idx)
            print(f"  -> '{selected_name}' adicionada.")

        except ValueError:
            print("  >> Entrada numérica inválida.")
        except KeyboardInterrupt:
             print("\nOperação cancelada.")
             return {} # Exit gracefully on Ctrl+C

    if not chosen_series_names: return {} # Handle case where user types 'fim' immediately

    print("\nPasso 2: Avalie as séries escolhidas (1-5).")
    for series_name in chosen_series_names:
         while True:
            prompt = f"  Nota para '{series_name}': "
            try:
                rating_str = input(prompt).strip()
                # Do not allow skipping here as user explicitly chose these
                if not rating_str: print("    >> Por favor, forneça uma nota."); continue
                rating = int(rating_str)
                if 1 <= rating <= 5: user_ratings[series_name] = rating; break
                else: print("    >> Nota inválida. Use um número de 1 a 5.")
            except ValueError:
                print("    >> Entrada inválida. Digite um número.")
            except KeyboardInterrupt:
                 print("\nOperação cancelada.")
                 return {}

    return user_ratings


def _print_recommendations(recommendations: list[Recommendation]) -> None:
    """Formats and prints the list of hybrid recommendations."""
    print("\n" + "=" * 25 + " Séries Recomendadas (Híbrido) " + "=" * 25)
    if not recommendations:
        print("Não foi possível gerar recomendações com as avaliações fornecidas.")
        return

    for position, rec in enumerate(recommendations, start=1):
        print(f"\n{position}. {rec.name}")
        print(f"   Score Final..: {rec.score:.3f}")
        # Show breakdown of scores if available
        if rec.content_score is not None and rec.collab_score is not None:
             print(f"   (Conteúdo...: {rec.content_score:.3f} | Colaborativo: {rec.collab_score:.3f})")
        # Display metadata
        meta_str = " | ".join(f"{k.replace('_',' ').capitalize()}: {v}" for k, v in rec.metadata.items() if v != 'N/A' and v != '')
        if meta_str:
             print(f"   Metadados....: {meta_str}")


def main() -> None:
    """Main execution function for the command-line hybrid recommender."""
    print("=" * 70)
    print(" Guia de Séries – Sistema de Recomendação Híbrido v3.0 ".center(70))
    print("=" * 70)

    name = input("\nPara começar, informe seu nome: ").strip() or "Visitante"

    try:
        # Initialize the recommender with BOTH data files
        recommender = ContentRecommender(PRODUCTS_PATH, RATINGS_PATH)
    except FileNotFoundError as e:
         print(f"\n❌ Erro Fatal: Arquivo não encontrado: {e.filename}")
         print(f"   Verifique se '{PRODUCTS_PATH.name}' e '{RATINGS_PATH.name}' existem em:")
         print(f"   '{DATA_DIR}'")
         return
    except (ValueError, KeyError) as error:
        print(f"\n❌ Erro ao carregar ou validar os dados: {error}")
        return
    except Exception as e:
         print(f"\n❌ Erro inesperado ao inicializar o sistema: {e}")
         return


    print(f"\nOlá, {name}! Bem-vindo(a).")
    user_ratings_dict = _collect_ratings_from_choice(recommender, num_to_choose=5)

    if not user_ratings_dict:
        print("\nNenhuma avaliação válida fornecida. Encerrando.")
        return

    print("\nCalculando seu perfil e buscando recomendações (híbrido)...")
    try:
        # Build profile returns BOTH content vector and user rating Series
        profile_vector, new_user_ratings_series = recommender.build_profile_from_ratings(user_ratings_dict)

        # Call the HYBRID recommend method
        recommendations = recommender.recommend(
            profile_content_vector=profile_vector,
            new_user_ratings=new_user_ratings_series,
            series_ja_avaliadas=list(user_ratings_dict.keys()),
            top_n=5,
            content_weight=0.6, # Example weight: 60% content, 40% collaborative
            k_neighbors_collab=15 # Example: Use top 15 similar users
        )
    except Exception as e:
         print(f"\n❌ Erro ao gerar recomendações: {e}")
         # import traceback; traceback.print_exc() # Uncomment for detailed error stack
         return

    _print_recommendations(recommendations)

    # Optional: Save recommendations to JSON
    export = input("\nDeseja salvar as recomendações em JSON? (s/N): ").strip().lower()
    if export == "s":
        output_file = Path("recomendacoes_hibridas.json")
        try:
            output_file.write_text(recommender.to_json(recommendations), encoding="utf-8")
            print(f"✅ Recomendações salvas em {output_file.resolve()}")
        except Exception as e:
            print(f"❌ Erro ao salvar o arquivo JSON: {e}")

    print("\n" + "=" * 70)
    print(" Sistema Finalizado ".center(70, "="))
    print("=" * 70)

if __name__ == "__main__":
    main()
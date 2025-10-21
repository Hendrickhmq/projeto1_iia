# Guia de Séries – Sistema de Recomendação Híbrido

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-blueviolet?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-gray?style=for-the-badge&logo=numpy)
![Scikit-learn](https://img.shields.io/badge/SciKit--Learn-orange?style=for-the-badge&logo=scikit-learn)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hendrickhmq/projeto1_iia/blob/main/projeto1_IIA.ipynb) 
*(Clique para abrir o projeto funcional no Google Colab)*

---

**Autores:**
-   Arthur Fernandes Vargas (231013171)
-   Hendrick Henrique Moreno Quevedo (231025510)

**Disciplina:** Introdução à Inteligência Artificial (CIC0135) - 2025/2  
**Professor:** Dibio Leandro Borges

---

## 🎯 Visão Geral do Projeto

Sistema de recomendação **híbrido** que combina duas técnicas clássicas de filtragem — **Baseada em Conteúdo (TF-IDF)** e **Colaborativa (User-User)** — para gerar sugestões personalizadas de séries para um novo usuário.

A abordagem híbrida foi escolhida para cumprir todos os requisitos do projeto: implementar um modelo de conteúdo (Passo 3) e, ao mesmo tempo, dar uma aplicação prática à matriz de utilidade de 500 avaliações (Passo 2), usando-a para o modelo colaborativo.

### Recursos Principais

- **Modelo de Conteúdo (TF-IDF):** Gera um "perfil de gosto" vetorial para o usuário com base em suas avaliações, usando uma **média ponderada** (notas altas "puxam", notas baixas "empurram" o perfil).
- **Modelo Colaborativo (User-User):** Utiliza a matriz de utilidade (`user_ratings.csv`) para encontrar "gêmeos de gosto" (vizinhos) e prever notas para itens que o usuário ainda não viu (k-NN).
- **Modelo Híbrido:** Combina os scores dos dois modelos (após normalização Min-Max) em um `final_score` ponderado, garantindo recomendações robustas.
- **Solução para "Cold Start":** O modelo de conteúdo atua como *fallback*, garantindo que o sistema possa recomendar itens para um novo usuário (cujas notas não batem com nenhum "vizinho") desde a sua primeira avaliação.
- **Interface Interativa:** Um notebook do Google Colab (`projeto1_IIA.ipynb`) usa `ipywidgets` para uma interface de usuário limpa, que coleta notas e exibe as recomendações.

## 🏃 Como Executar (Recomendado)

A forma mais fácil de executar o projeto é através do Google Colab:

1.  Clique no badge "Open in Colab" no topo deste README.
2.  No Colab, clique em "Ambiente de execução" -> "Executar tudo".
3.  Os arquivos CSV serão baixados automaticamente do GitHub (com um *fallback* para upload manual).
4.  Role até a última célula de "Interface Interativa" para usar o sistema.

---

### Execução Local (Alternativa)

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Hendrickhmq/projeto1_iia.git](https://github.com/Hendrickhmq/projeto1_iia.git)
    cd projeto1_iia
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # Linux/macOS
    source .venv/bin/activate
    # Windows
    .\.venv\Scripts\activate
    ```
3.  **Instale as dependências:**
    ```bash
    python -m pip install -r requirements.txt
    ```
4.  **Execute o notebook Jupyter:**
    ```bash
    jupyter notebook projeto1_IIA.ipynb
    ```

## ⚙️ Como o Recomendador Híbrido Funciona

O sistema opera em três estágios para gerar uma recomendação para um novo usuário:

1.  **Estágio 1: Modelo de Conteúdo (TF-IDF)**
    - O catálogo de séries (`products.csv`) é vetorizado usando `TfidfVectorizer` (do `sklearn`), criando uma matriz de features.
    - As notas (1-5) que o novo usuário fornece são usadas para criar um **vetor de perfil** (`np.ndarray`).
    - O `content_score` é a **Similaridade de Cosseno** (`sklearn.metrics.cosine_similarity`) entre o vetor de perfil do usuário e os vetores de todas as outras séries.

2.  **Estágio 2: Modelo Colaborativo (User-User)**
    - A matriz de utilidade (`user_ratings.csv`) é pivotada para `usuários` x `itens`.
    - A similaridade de cosseno (`sklearn.metrics.pairwise_distances`) é calculada entre o *vetor de notas* do novo usuário e os vetores de todos os 500 usuários antigos.
    - O sistema encontra os "Top-K" vizinhos (`k_neighbors_collab`) mais parecidos.
    - O `collab_score` é uma **nota prevista** para cada série, calculada pela média ponderada das notas dadas pelos vizinhos.

3.  **Estágio 3: O Modelo Híbrido**
    - Os `content_score` e `collab_score` são normalizados (escala de 0 a 1).
    - Um `final_score` é calculado como uma média ponderada dos dois scores (ex: 60% Conteúdo, 40% Colaborativo).
    - As Top-N séries com o maior `final_score` são retornadas ao usuário.

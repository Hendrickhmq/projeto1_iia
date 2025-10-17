# Mercado Verde Viva – Sistema de Recomendação

Este projeto implementa um sistema de recomendação baseado em conteúdo para uma loja de produtos naturais chamada **Mercado Verde Viva**. O catálogo possui 20 produtos, cada um descrito por três características principais (categoria, sabor predominante e benefício). Uma matriz de utilidade com 500 interações usuário-produto acompanha o repositório para referência.

## Conteúdo

- `content_recommender/data/products.csv`: catálogo com 20 produtos e três características cada.
- `content_recommender/data/user_ratings.csv`: 500 avaliações simuladas (escala 1-5) entre usuários e produtos.
- `content_recommender/recommender.py`: implementação do recomendador utilizando TF-IDF para extrair as características mais relevantes.
- `app.py`: interface em linha de comando para cadastro rápido de preferências e geração de recomendações personalizadas.

## Pré-requisitos

- Python 3.10+
- Dependências listadas em `requirements.txt`.

## Como executar

1. Crie um ambiente virtual opcionalmente:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\\Scripts\\activate   # Windows
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Inicie a aplicação:
   ```bash
   python app.py
   ```
4. Informe seu nome, selecione pelo menos uma preferência entre categoria, sabor e benefício, e receba as recomendações.

Ao final, é possível exportar as sugestões em formato JSON para consulta posterior.

## Como funciona o modelo

1. As características textuais dos produtos são combinadas e vetorizadas com **TF-IDF**.
2. Um novo usuário informa preferências para as três dimensões do catálogo.
3. O perfil textual do usuário é convertido para o mesmo espaço vetorial e a similaridade cosseno é usada para ranquear os produtos.
4. Os cinco itens mais similares são exibidos na interface.

A matriz de utilidade (avaliações) está incluída para consultas futuras e pode ser utilizada para validar outras abordagens de recomendação.

# Metro Anomaly Dashboard App

Aplicação em formato parecido com o modelo da imagem:

```text
metro-anomaly-dashboard-app/
├── backend/
│   ├── __init__.py
│   └── main.py
├── frontend/
│   └── index.html
├── models/
│   ├── resultados_anomalias.csv
│   ├── modelo_isolation_forest.pkl
│   ├── preprocessor.pkl
│   └── features_modelo.pkl
├── Dockerfile
├── README.md
└── requirements.txt
```

## O que cada pasta faz

- `backend/`: API em FastAPI. Lê os resultados do modelo, aplica filtros e envia dados para o front-end.
- `frontend/`: tela do dashboard em HTML, CSS e JavaScript.
- `models/`: arquivos gerados pelo notebook.
- `Dockerfile`: usado para publicar no Render ou Railway.

## Arquivo obrigatório

O arquivo mais importante é:

```text
models/resultados_anomalias.csv
```

Ele deve ser gerado pelo notebook com:

```python
df_resultado.to_csv("resultados_anomalias.csv", index=False)
```

Depois copie esse arquivo para dentro da pasta `models/`.

## Arquivos opcionais do modelo

Estes arquivos são úteis para o endpoint `/api/predict`, mas o dashboard funciona principalmente com o CSV:

```text
models/modelo_isolation_forest.pkl
models/preprocessor.pkl
models/features_modelo.pkl
```

Eles são gerados no notebook com:

```python
joblib.dump(modelo_iforest, "modelo_isolation_forest.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(features_modelo, "features_modelo.pkl")
```

## Rodar localmente

Instale as dependências:

```bash
pip install -r requirements.txt
```

Rode o backend:

```bash
uvicorn backend.main:app --reload
```

Abra no navegador:

```text
http://127.0.0.1:8000
```

## Publicar no Render

Crie um Web Service usando Docker.

O Render vai usar automaticamente o `Dockerfile`.

Depois do deploy, use o link público no relatório.

## Publicar no Railway

Crie um projeto a partir do repositório GitHub.

O Railway também consegue usar o `Dockerfile`.

## Observação importante

O dashboard foi feito para apresentação do trabalho de Mineração de Dados. O notebook treina o modelo e exporta os resultados. A aplicação consome esses resultados e apresenta:

- descrição do projeto;
- informações do dataset;
- gráficos exploratórios;
- quantidade e proporção de anomalias;
- anomaly score;
- tabela de registros anômalos;
- filtros;
- interpretação prática;
- conclusão.

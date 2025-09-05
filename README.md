# Predição de Performance de Criativos em Tráfego Pago usando Deep Learning

## Descrição

Este projeto tem como objetivo desenvolver um modelo que prevê a performance de publicações pagas em redes sociais, utilizando análise de imagens dos criativos e dados históricos de campanhas anteriores. O foco é identificar quais imagens dos produtos têm maior potencial para gerar melhor retorno sobre investimento (ROI), otimizando assim o uso de recursos em mídia paga.

O sistema combina técnicas de deep learning para análise visual, dados temporais e metadados de campanhas para gerar predições de métricas como CTR, taxa de conversão e ROAS, além de sugerir os melhores horários para veiculação dos anúncios.

## Estrutura do Projeto

Project-root/
 │  
 ├── data/  
 │ ├── raw/                  # Dados brutos, imagens originais e metadados não processados  
 │ │ ├── for_fit/            # Dados para treino e teste do modelo  
 │ │ └── for_predict/        # Imagens a serem previstas  
 │ └── processed/            # Dados pré-processados prontos para uso (matrizes, features)  
 │  
 ├── notebooks/              # Jupyter notebooks para exploração, análise e testes  
 │  
 ├── src/                    # Código-fonte do projeto  
 │ ├── data_preprocessing/   # Scripts para carregamento e limpeza dos dados, pré-processamento de imagens  
 │ ├── feature_extraction/   # Scripts para extração de features visuais e textuais  
 │ ├── models/               # Definição e treinamento dos modelos de deep learning  
 │ ├── evaluation/           # Scripts para avaliação de métricas e análise dos  
 │ │	resultados  
 │ ├── utils/                # Funções utilitárias e helpers gerais  
 │ └── inference/            # Código para gerar predições com modelos treinados  
 │  
 ├── experiments/            # Configurações, logs e resultados de experimentos de modelagem  
 │  
 ├── reports/                # Documentação do projeto, relatórios e apresentações  
 │  
 ├── requirements.txt        # Lista de dependências necessárias para rodar o projeto  
 ├── README.md               # Apresentação do projeto e instruções principais  
 └── .gitignore              # Arquivos e pastas ignorados pelo Git  
  
## Como Usar

1. Instale as dependências necessárias listadas em `requirements.txt`;
2. Prepare os dados para ajuste do modelo na pasta `data/raw/`;
3. Prepare as imagens, cujas métricas se deseja, prever na pasta `data/raw/`;
4. Ajuste os parâmetros iniciais do arquivo `main.ipynb` na pasta `notebooks/` e execute-o;
5. Avalie os resultados no próprio arquivo `main.ipynb` ou dentre os relatórios em `reoprts/`;

## Dependências

- Python 3.11+
- TensorFlow
- pandas, numpy, scikit-learn
- OpenCV ou PIL para processamento de imagens
- Jupyter para notebooks

## Autores

- Felipe Neres Silva Bezerra
- Guilherme Da Silva Brevilato
- Isabela Kazue Sinoduka
- Pedro Henrique Bittencourt De Freitas 

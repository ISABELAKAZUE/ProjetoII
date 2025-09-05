# Predição de Performance de Criativos em Tráfego Pago usando Deep Learning

## Descrição

Este projeto tem como objetivo desenvolver um modelo que prevê a performance de publicações pagas em redes sociais, utilizando análise de imagens dos criativos e dados históricos de campanhas anteriores. O foco é identificar quais imagens dos produtos têm maior potencial para gerar melhor retorno sobre investimento (ROI), otimizando assim o uso de recursos em mídia paga.

O sistema combina técnicas de deep learning para análise visual, dados temporais e metadados de campanhas para gerar predições de métricas como CTR, taxa de conversão e ROAS, além de sugerir os melhores horários para veiculação dos anúncios.

## Estrutura do Projeto

- `data/` - Dados brutos e processados (imagens e métricas)
- `notebooks/` - Notebooks para exploração e análises preliminares
- `src/` - Código-fonte para pré-processamento, modelagem, avaliação e inferência
- `reports/` - Documentação e relatórios do projeto
- `experiments/` - Resultados e logs dos experimentos realizados

## Como Usar

1. Instale as dependências necessárias listadas em `requirements.txt`.
2. Prepare os dados brutos na pasta `data/raw/`.
3. Execute os scripts de pré-processamento para gerar os dados prontos para modelagem.
4. Treine os modelos usando os scripts em `src/models/`.
5. Avalie a performance com os scripts em `src/evaluation/`.
6. Use o código em `src/inference/` para testar novos criativos.

## Dependências

- Python 3.11+
- TensorFlow ou PyTorch (especificar qual foi usado)
- pandas, numpy, scikit-learn
- OpenCV ou PIL para processamento de imagens
- Jupyter para notebooks

## Contato

Autores:
- Felipe Neres Silva Bezerra
- Guilherme Da Silva Brevilato
- Isabela Kazue Sinoduka
- Pedro Henrique Bittencourt De Freitas 

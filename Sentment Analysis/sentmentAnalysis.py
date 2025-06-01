import pandas as pd

# Criando os dados para o DataFrame
dados = {'Respostas': [
    "Amei o produto, excelente qualidade!",
    "Não gostei muito, esperava mais.",
    "É razoável, cumpre o que promete.",
    "Péssima experiência, não recomendo.",
    "Muito bom, superou minhas expectativas!",
    "Normal, nada de especial.",
    "Horrível, dinheiro jogado fora.",
    "Gostei bastante, voltarei a comprar.",
    "Mais ou menos, pelo preço poderia ser melhor."
]}

# Criando o DataFrame
df = pd.DataFrame(dados)

from transformers import pipeline, XLMRobertaTokenizer

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def estrelas_para_sentimento(label_estrelas):
    estrelas = int(label_estrelas.split()[0])
    if estrelas <= 2:
        return "Negativo"
    elif estrelas == 3:
        return "Neutro"
    else:
        return "Positivo"

sentiments = []  
for resposta in df["Respostas"]:
    result = sentiment_pipeline(resposta[:512])[0] 
    sentimento = estrelas_para_sentimento(result['label'])
    sentiments.append(sentimento)

df["Sentimento"] = sentiments

print(df[["Respostas", "Sentimento"]])
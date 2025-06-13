# arquivo: cliente_processador.py
import pandas as pd
import requests # Para fazer chamadas HTTP
from datetime import datetime

# Importe as classes e funções que já criamos
from transformers import pipeline
from bertopic import BERTopic

class ProcessadorDeComentarios:
    """
    Carrega modelos de NLP, processa um DataFrame e envia os resultados
    para uma API externa.
    """
    def __init__(self, api_url: str):
        if not api_url.endswith('/'):
            api_url += '/'
        self.api_url = api_url
        self._carregar_modelos()

    def _carregar_modelos(self):
        """Método privado para carregar todos os modelos necessários."""
        print("Cliente: Carregando modelos de NLP...")
        self.pipeline_sentimento = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        self.pipeline_ner = pipeline("ner", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese", aggregation_strategy="simple")
        
        # Modelo Zero-shot para classificação flexível
        self.pipeline_classificador = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.categorias_candidatas = ["Pagamento", "App com Bug", "Atendimento", "Sugestão"]
        
        # Carregando modelo BERTopic (usando um pré-treinado como exemplo)
        self.modelo_topico = BERTopic.load("MaartenGr/BERTopic_ArXiv")
        print("Cliente: Modelos carregados.")

    @staticmethod
    def _estrelas_para_sentimento(label: str, score: float) -> dict:
        estrelas = int(label[0])
        if estrelas <= 2: sentimento_label = "Negativo"
        elif estrelas == 3: sentimento_label = "Neutro"
        else: sentimento_label = "Positivo"
        return {"label": sentimento_label, "score": score}

    def _criar_payload_analise(self, texto: str) -> dict:
        """Executa todos os modelos em um único texto e monta o JSON."""
        # 1. Sentimento
        res_sent = self.pipeline_sentimento(texto)[0]
        sentimento = self._estrelas_para_sentimento(res_sent['label'], res_sent['score'])

        # 2. Classificação
        res_class = self.pipeline_classificador(texto, candidate_labels=self.categorias_candidatas)
        classificacao = {"categoria": res_class['labels'][0], "score": res_class['scores'][0]}
        
        # 3. NER
        entidades = [{"entidade": e['word'], "tipo": e['entity_group'], "score": e['score']} for e in self.pipeline_ner(texto)]

        # 4. Tópico
        topic_num, _ = self.modelo_topico.transform([texto])
        topic_info = self.modelo_topico.get_topic_info(topic_num[0])
        topico = {"id_topico": topic_num[0], "nome_topico": topic_info['Name'].iloc[0]}

        # Monta o payload final
        payload = {
            "texto_original": texto,
            "sentimento": sentimento,
            "classificacao_primaria": classificacao,
            "topico_detectado": topico,
            "entidades_extraidas": entidades,
            "timestamp_analise": datetime.utcnow().isoformat() + "Z"
        }
        return payload

    def processar_e_enviar(self, df: pd.DataFrame, coluna_texto: str):
        """Itera sobre o DataFrame, processa cada texto e envia para a API."""
        print(f"\nCliente: Iniciando processamento de {len(df)} comentários...")
        for index, row in df.iterrows():
            texto = row[coluna_texto]
            if not isinstance(texto, str) or not texto.strip():
                print(f"Cliente: Linha {index} ignorada (texto inválido).")
                continue

            print(f"Cliente: Processando linha {index}...")
            # 1. Cria o payload JSON com a análise completa
            payload_final = self._criar_payload_analise(texto)
            
            # 2. Envia para a API
            try:
                endpoint = self.api_url + "receber_analise"
                response = requests.post(endpoint, json=payload_final, timeout=60) # Timeout de 60s
                
                if response.status_code == 201:
                    print(f"Cliente: Linha {index} enviada com sucesso! Resposta: {response.json()}")
                else:
                    print(f"Cliente: Erro ao enviar linha {index}. Status: {response.status_code}, Resposta: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Cliente: Falha de conexão ao enviar linha {index}. Erro: {e}")
        
        print("Cliente: Processamento concluído.")

# arquivo: servidor_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Modelos de Dados (Contrato da API) ---
class Sentimento(BaseModel):
    label: str
    score: float

class Classificacao(BaseModel):
    categoria: str
    score: float

class Entidade(BaseModel):
    entidade: str
    tipo: str
    score: float

class Topico(BaseModel):
    id_topico: int
    nome_topico: str

class AnaliseCompleta(BaseModel):
    """Este é o JSON que nossa API espera receber."""
    texto_original: str
    sentimento: Sentimento
    classificacao_primaria: Classificacao
    topico_detectado: Topico
    entidades_extraidas: List[Entidade]
    timestamp_analise: str = Field(..., example="2025-06-12T22:20:13.123Z")


# --- Banco de Dados em Memória (Para a Demonstração) ---
# Em um projeto real, isso seria um banco de dados como PostgreSQL, MongoDB, etc.
db_analises: List[AnaliseCompleta] = []


# --- Aplicação FastAPI ---
app = FastAPI(
    title="API de Coleta de Análises",
    description="Recebe e armazena análises de sentimento processadas por clientes."
)

@app.post("/receber_analise", status_code=201)
def receber_analise(analise: AnaliseCompleta):
    """
    Endpoint para receber uma análise completa e 'salvá-la' no nosso banco de dados.
    """
    print(f"Recebida análise para o texto: '{analise.texto_original[:50]}...'")
    db_analises.append(analise)
    return {"status": "sucesso", "total_armazenado": len(db_analises)}

@app.get("/ver_analises", response_model=List[AnaliseCompleta])
def ver_analises():
    """Endpoint para visualizar todas as análises recebidas."""
    if not db_analises:
        raise HTTPException(status_code=404, detail="Nenhuma análise foi recebida ainda.")
    return db_analises

print("Servidor pronto. Acesse http://127.0.0.1:8000/docs para ver a documentação.")
import pandas as pd
from cliente_processador import ProcessadorDeComentarios

if __name__ == "__main__":
    # URL do seu servidor FastAPI
    API_URL = "http://127.0.0.1:8000"

    # DataFrame de exemplo
    dados = {
        "Respostas": [
            "Amei o novo recurso de pagamento com Pix, mas o motorista demorou muito.",
            "Que app horrível, travou na hora de fechar a corrida e cobrou duas vezes!",
            "O atendente João foi muito solícito e resolveu meu problema rapidamente."
        ]
    }
    df_exemplo = pd.DataFrame(dados)

    # Instancia e executa o processador
    processador = ProcessadorDeComentarios(api_url=API_URL)
    processador.processar_e_enviar(df_exemplo, "Respostas")
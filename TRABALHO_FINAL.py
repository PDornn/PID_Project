#%pip install ultralytics opencv-python-headless matplotlib requests

import cv2
import os
import requests
import numpy as np
import matplotlib.pyplot as plt


from ultralytics import YOLO
from pathlib import Path

# Exemplo de criação da estrutura de pastas exigida pelo documento
os.makedirs("PID_Project/imagens/originais", exist_ok=True)
os.makedirs("PID_Project/imagens/processadas", exist_ok=True)
os.makedirs("PID_Project/modelos", exist_ok=True)

plt.rcParams['figure.figsize'] = [10, 8]
IMAGENS_ORIGINAIS = Path('PID_Project/imagens/originais/')
IMAGENS_PROCESSADAS = Path('PID_Project/imagens/processadas/')



print("Estrutura de diretórios criada com sucesso!")

def pre_processamento(caminho_imagem):
    # Carregar imagem original (O OpenCV lê em BGR)
    img = cv2.imread(str(caminho_imagem))
    if img is None:
        print(f"⚠️ Erro ao carregar a imagem: {caminho_imagem}")
        return None, None, None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Aplicar Blur para suavizar ruídos (Filtro Gaussiano)
    img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    # 2. Conversão para HSV para isolar as manchas/doenças (Tons de marrom/amarelo)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definição de thresholds de cor (Tons amarelados/castanhos da doença)
    limite_inferior = np.array([10, 50, 50])
    limite_superior = np.array([30, 255, 255])

    # Criar máscara isolando apenas a região da doença
    mascara = cv2.inRange(img_hsv, limite_inferior, limite_superior)
    resultado_threshold = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara)

    # Retorna as 3 variáveis que a função de lote espera receber
    return img_rgb, img_blur, resultado_threshold

def carregar_e_processar_pasta(pasta_origem, pasta_destino):
    """
    Varre a pasta de origem, aplica o pré-processamento e salva
    fisicamente os resultados na Path de destino.
    """
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    # Garante que a pasta de destino exista no Google Drive
    pasta_destino.mkdir(parents=True, exist_ok=True)

    # Listar e ordenar usando a biblioteca Path
    arquivos = sorted(pasta_origem.iterdir())
    imagens_carregadas = []

    print(f"🔍 Verificando a pasta de origem: '{pasta_origem}'")

    for caminho_arquivo in arquivos:
        # Verifica se é um arquivo de imagem válido
        if caminho_arquivo.suffix.lower() in extensoes_validas:
            nome_arquivo = caminho_arquivo.name
            print(f"📸 Processando: {nome_arquivo}")

            # Executa o pré-processamento passando o caminho completo (Path)
            img_original, img_blur, img_thresh = pre_processamento(caminho_arquivo)
            
            if img_original is None:
                continue

            # --- SALVAMENTO FISICO ---
            # O OpenCV precisa receber a imagem de volta em BGR para salvar corretamente
            img_thresh_bgr = cv2.cvtColor(img_thresh, cv2.COLOR_RGB2BGR)
            caminho_salvamento = pasta_destino / f"proc_{nome_arquivo}"
            cv2.imwrite(str(caminho_salvamento), img_thresh_bgr)
            # -----------------------------------------

            # Armazena os resultados na memória (lista) para uso posterior no Colab
            dados_imagem = {
                "nome": nome_arquivo,
                "original": img_original,
                "blur": img_blur,
                "threshold": img_thresh,
                "caminho_salvo": caminho_salvamento
            }
            imagens_carregadas.append(dados_imagem)

    print(f"\n Concluído! {len(imagens_carregadas)} imagens processadas e salvas em: '{pasta_destino}'")
    return imagens_carregadas

lote_de_imagens = carregar_e_processar_pasta(IMAGENS_ORIGINAIS, IMAGENS_PROCESSADAS)

#%pip install ultralytics opencv-python-headless matplotlib requests

import cv2
import os
import requests
import numpy as np
import matplotlib.pyplot as plt


from ultralytics import YOLO
from pathlib import Path



plt.rcParams['figure.figsize'] = [10, 8]
IMAGENS_ORIGINAIS = Path('PID_Project/imagens/originais/')
IMAGENS_PROCESSADAS = Path('PID_Project/imagens/processadas/')

# Exemplo de criação da estrutura de pastas exigida pelo documento
os.makedirs("PID_Project/imagens/originais", exist_ok=True)
os.makedirs("PID_Project/imagens/processadas", exist_ok=True)
os.makedirs("PID_Project/modelos", exist_ok=True)


def carregar_imagens(diretorio):
    """Carrega todas as imagens da pasta origem e retorna uma lista."""
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    lista_imagens = []
    
    for arquivo in diretorio.iterdir():
        # Verifica se a extensão do arquivo é válida
        if arquivo.suffix.lower() in extensoes:
            img = cv2.imread(str(arquivo))
            if img is not None:
                # Salva o nome do arquivo e a imagem em uma lista
                lista_imagens.append([arquivo.name, img])
                print("Imagem carregada:", arquivo.name)
                
    return lista_imagens

def pre_processar_imagem(img):
    """Aplica escala de cinza, desfoque e binarização em uma imagem."""
    # 1. Escala de Cinza
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Filtro de Suavização (Gaussian Blur)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # 3. Limiarização (Threshold de Otsu)
    ret, binarizada = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Retorna as três imagens geradas
    return cinza, suavizada, binarizada

def salvar_imagem(nome_original, imagem, sufixo_etapa, diretorio_destino):
    """
    Salva uma imagem qualquer adicionando um sufixo ao nome original.
    Exemplo: "imagem1.jpg" com sufixo "canny" vira "canny_imagem1.jpg".
    """
    # Cria o nome do arquivo combinando o sufixo e o nome original
    nome_arquivo_final = sufixo_etapa + "_" + nome_original
    caminho_completo = diretorio_destino / nome_arquivo_final
    
    # Grava o arquivo no disco
    cv2.imwrite(str(caminho_completo), imagem)
    print("Arquivo salvo:", nome_arquivo_final)


# --- Execução do Fluxo com o Novo Método ---

# 1. Carrega as imagens
imagens_originais = carregar_imagens(IMAGENS_ORIGINAIS)

# Lista para guardar temporariamente os resultados na memória
resultados_processados = []

# 2. Processa
for item in imagens_originais:
    nome = item[0]
    img = item[1]
    
    cinza, suavizada, binarizada = pre_processar_imagem(img)
    resultados_processados.append([nome, cinza, suavizada, binarizada])

# 3. Salva utilizando o método genérico para cada etapa individualmente
for resultado in resultados_processados:
    nome = resultado[0]
    img_cinza = resultado[1]
    img_suavizada = resultado[2]
    img_binarizada = resultado[3]
    
    # Chamamos a mesma função mudando apenas o sufixo correspondente
    salvar_imagem(nome, img_cinza, "cinza", IMAGENS_PROCESSADAS)
    salvar_imagem(nome, img_suavizada, "suavizada", IMAGENS_PROCESSADAS)
    salvar_imagem(nome, img_binarizada, "binarizada", IMAGENS_PROCESSADAS)

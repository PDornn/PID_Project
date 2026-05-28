"""
Dataset Import - PlantVillage
Capstone: Processamento Digital de Imagens
Dataset: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Estrutura esperada após download:
  data/PlantVillage/
    ├── Apple___Apple_scab/
    ├── Apple___healthy/
    ├── Tomato___Early_blight/
    └── ...  (38 classes no total)
"""

import cv2
import os
import random
from pathlib import Path

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS = os.path.join(PROJETO_RAIZ, 'data')
PASTA_DATASET = os.path.join(PASTA_DADOS, 'PlantVillage')

_EXTENSOES_VALIDAS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def verificar_dataset(caminho=None):
    """Verifica se o dataset PlantVillage está disponível no disco."""
    pasta = caminho or PASTA_DATASET
    if not os.path.exists(pasta):
        print(f"[Dataset] Pasta nao encontrada: {pasta}")
        print("[Dataset] Baixe o dataset em:")
        print("  https://www.kaggle.com/datasets/mohitsingh1804/plantvillage")
        print(f"[Dataset] Extraia em: {PASTA_DADOS}/PlantVillage/")
        return False
    classes = listar_classes(pasta)
    if not classes:
        print(f"[Dataset] Nenhuma classe encontrada em: {pasta}")
        return False
    print(f"[Dataset] OK — {len(classes)} classes em {pasta}")
    return True


def listar_classes(caminho=None):
    """Retorna lista ordenada de nomes de classes (subpastas do dataset)."""
    pasta = caminho or PASTA_DATASET
    if not os.path.exists(pasta):
        return []
    return sorted([
        d for d in os.listdir(pasta)
        if os.path.isdir(os.path.join(pasta, d))
    ])


def _listar_imagens(pasta):
    """Lista caminhos de todos os arquivos de imagem em uma pasta."""
    if not os.path.exists(pasta):
        return []
    return [
        os.path.join(pasta, f)
        for f in os.listdir(pasta)
        if Path(f).suffix.lower() in _EXTENSOES_VALIDAS
    ]


def estatisticas_dataset(caminho=None):
    """
    Calcula e exibe estatísticas do dataset.

    Retorna dict com:
      - n_classes       : número de classes
      - total_imagens   : total de imagens
      - por_classe      : {nome_classe: contagem}
      - saudaveis       : lista de classes sem doença
      - doentes         : lista de classes com doença
    """
    pasta = caminho or PASTA_DATASET
    classes = listar_classes(pasta)
    por_classe = {}
    total = 0

    for cls in classes:
        imgs = _listar_imagens(os.path.join(pasta, cls))
        por_classe[cls] = len(imgs)
        total += len(imgs)

    saudaveis, doentes = separar_saudaveis_doentes(classes)

    return {
        'n_classes': len(classes),
        'total_imagens': total,
        'por_classe': por_classe,
        'saudaveis': saudaveis,
        'doentes': doentes,
    }


def carregar_imagem(caminho, tamanho=None):
    """
    Carrega imagem do disco em formato RGB (numpy uint8).

    Parâmetros:
        caminho - caminho completo do arquivo
        tamanho - (largura, altura) para redimensionar; None mantém original
    """
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if tamanho is not None:
        img = cv2.resize(img, tamanho)
    return img


def amostrar_por_classe(n=1, classe=None, caminho=None, semente=42):
    """
    Retorna n amostras aleatórias do dataset.

    Parâmetros:
        n       - amostras por classe (ou total, se classe informada)
        classe  - nome da classe específica; None para todas
        caminho - raiz do dataset
        semente - para reprodutibilidade

    Retorna:
        lista de dicts: {'caminho', 'classe', 'img'}
    """
    random.seed(semente)
    pasta = caminho or PASTA_DATASET
    classes = [classe] if classe else listar_classes(pasta)
    amostras = []

    for cls in classes:
        imagens = _listar_imagens(os.path.join(pasta, cls))
        if not imagens:
            continue
        selecionadas = random.sample(imagens, min(n, len(imagens)))
        for p in selecionadas:
            amostras.append({
                'caminho': p,
                'classe': cls,
                'img': carregar_imagem(p),
            })

    return amostras


def amostrar_pares(plantas=None, n=1, caminho=None, semente=42):
    """
    Retorna pares (saudável, doente) para cada planta.

    Parâmetros:
        plantas - lista de nomes de plantas (ex: ['Tomato', 'Potato']);
                  None para todas as plantas disponíveis
        n       - número de amostras por classe
        caminho - raiz do dataset
        semente - semente aleatória

    Retorna:
        lista de dicts: {'planta', 'classe_saudavel', 'classe_doente',
                         'img_saudavel', 'img_doente'}
    """
    random.seed(semente)
    pasta = caminho or PASTA_DATASET
    classes = listar_classes(pasta)
    saudaveis, doentes = separar_saudaveis_doentes(classes)

    mapa_saudavel = {_planta_de_classe(c): c for c in saudaveis}
    mapa_doente = {}
    for c in doentes:
        p = _planta_de_classe(c)
        mapa_doente.setdefault(p, []).append(c)

    if plantas is None:
        plantas = sorted(set(mapa_saudavel) & set(mapa_doente))

    pares = []
    for planta in plantas:
        if planta not in mapa_saudavel or planta not in mapa_doente:
            continue
        cls_s = mapa_saudavel[planta]
        imgs_s = _listar_imagens(os.path.join(pasta, cls_s))
        if not imgs_s:
            continue

        for cls_d in mapa_doente[planta][:n]:
            imgs_d = _listar_imagens(os.path.join(pasta, cls_d))
            if not imgs_d:
                continue
            p_s = random.choice(imgs_s)
            p_d = random.choice(imgs_d)
            pares.append({
                'planta': planta,
                'classe_saudavel': cls_s,
                'classe_doente': cls_d,
                'img_saudavel': carregar_imagem(p_s),
                'img_doente': carregar_imagem(p_d),
            })

    return pares


def separar_saudaveis_doentes(classes):
    """
    Divide classes em saudáveis e doentes com base no sufixo 'healthy'.

    Retorna (saudaveis, doentes) como listas de strings.
    """
    saudaveis = [c for c in classes if 'healthy' in c.lower()]
    doentes = [c for c in classes if 'healthy' not in c.lower()]
    return saudaveis, doentes


def _planta_de_classe(nome_classe):
    """
    Extrai o nome da planta de um nome de classe do PlantVillage.
    Ex: 'Tomato___Early_blight' -> 'Tomato'
        'Apple___Apple_scab'    -> 'Apple'
    """
    separadores = ['___', '__', '_']
    for sep in separadores:
        if sep in nome_classe:
            return nome_classe.split(sep)[0]
    return nome_classe


def nome_legivel(nome_classe):
    """
    Converte nome técnico da classe em texto legível.
    Ex: 'Tomato___Early_blight' -> 'Tomato - Early Blight'
    """
    nome = nome_classe.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
    return nome.title()

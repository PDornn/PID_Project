"""
Dataset Import - PlantVillage
Capstone: Processamento Digital de Imagens
Dataset: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Estrutura do dataset (Kaggle):
  data/PlantVillage/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___healthy/
    │   ├── Tomato___Early_blight/
    │   └── ...  (38 classes)
    └── val/
        ├── Apple___Apple_scab/
        └── ...
"""

import cv2
import os
import random
from pathlib import Path

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS  = os.path.join(PROJETO_RAIZ, 'data')
PASTA_DATASET = os.path.join(PASTA_DADOS, 'PlantVillage')

_EXTENSOES_VALIDAS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ─── Utilitário de estrutura ─────────────────────────────────────────────────

def _pasta_split(caminho, split='train'):
    """
    Retorna a pasta que contém as subpastas de classes.

    Suporta duas estruturas:
      Split : PlantVillage/train/<classe>/  (padrão deste dataset Kaggle)
      Plana : PlantVillage/<classe>/
    """
    candidata = os.path.join(caminho, split)
    return candidata if os.path.isdir(candidata) else caminho


# ─── Verificação e metadados ─────────────────────────────────────────────────

def verificar_dataset(caminho=None):
    """Verifica se o dataset PlantVillage está disponível e imprime um resumo."""
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

    tem_split = os.path.isdir(os.path.join(pasta, 'train'))
    if tem_split:
        n_val = len(listar_classes(pasta, split='val'))
        print(f"[Dataset] OK — {len(classes)} classes  |  train + val")
        print(f"[Dataset]   train : {_pasta_split(pasta, 'train')}")
        if n_val:
            print(f"[Dataset]   val   : {_pasta_split(pasta, 'val')} ({n_val} classes)")
    else:
        print(f"[Dataset] OK — {len(classes)} classes em {pasta}")

    return True


def listar_classes(caminho=None, split='train'):
    """
    Retorna lista ordenada de classes do split indicado.

    Parâmetros:
        caminho - raiz do dataset (default: PASTA_DATASET)
        split   - 'train' ou 'val' (ignorado em estrutura plana)
    """
    pasta = _pasta_split(caminho or PASTA_DATASET, split)
    if not os.path.exists(pasta):
        return []
    return sorted(
        d for d in os.listdir(pasta)
        if os.path.isdir(os.path.join(pasta, d))
    )


def estatisticas_dataset(caminho=None):
    """
    Retorna estatísticas do dataset.

    Dict de retorno:
      n_classes      : número de classes
      total_train    : imagens em train
      total_val      : imagens em val (0 se não existir)
      por_classe     : {classe: contagem_train}
      por_classe_val : {classe: contagem_val}
      saudaveis      : lista de classes saudáveis
      doentes        : lista de classes com doença
    """
    pasta = caminho or PASTA_DATASET
    classes = listar_classes(pasta, split='train')

    por_classe, total_train = _contar_por_classe(pasta, classes, 'train')
    por_classe_val, total_val = _contar_por_classe(pasta, classes, 'val')

    saudaveis, doentes = separar_saudaveis_doentes(classes)

    return {
        'n_classes'     : len(classes),
        'total_train'   : total_train,
        'total_val'     : total_val,
        'total_imagens' : total_train,   # alias mantido para compatibilidade
        'por_classe'    : por_classe,
        'por_classe_val': por_classe_val,
        'saudaveis'     : saudaveis,
        'doentes'       : doentes,
    }


def _contar_por_classe(pasta_base, classes, split):
    """Conta imagens por classe no split dado. Retorna (dict, total)."""
    pasta = _pasta_split(pasta_base, split)
    por_classe = {}
    total = 0
    for cls in classes:
        n = len(_listar_imagens(os.path.join(pasta, cls)))
        por_classe[cls] = n
        total += n
    return por_classe, total


# ─── I/O de imagens ──────────────────────────────────────────────────────────

def _listar_imagens(pasta):
    """Lista caminhos de todos os arquivos de imagem em uma pasta."""
    if not os.path.exists(pasta):
        return []
    return [
        os.path.join(pasta, f)
        for f in os.listdir(pasta)
        if Path(f).suffix.lower() in _EXTENSOES_VALIDAS
    ]


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


# ─── Amostragem ──────────────────────────────────────────────────────────────

def amostrar_por_classe(n=1, classe=None, caminho=None, split='train', semente=42):
    """
    Retorna n amostras aleatórias por classe do split indicado.

    Parâmetros:
        n       - amostras por classe
        classe  - nome de uma classe específica; None para todas
        caminho - raiz do dataset
        split   - 'train' ou 'val'
        semente - para reprodutibilidade

    Retorna lista de dicts: {'caminho', 'classe', 'img'}
    """
    random.seed(semente)
    pasta = _pasta_split(caminho or PASTA_DATASET, split)
    classes = [classe] if classe else listar_classes(caminho, split=split)
    amostras = []

    for cls in classes:
        imagens = _listar_imagens(os.path.join(pasta, cls))
        if not imagens:
            continue
        for p in random.sample(imagens, min(n, len(imagens))):
            amostras.append({'caminho': p, 'classe': cls, 'img': carregar_imagem(p)})

    return amostras


def amostrar_pares(plantas=None, n=1, caminho=None, split='train', semente=42):
    """
    Retorna pares (saudável, doente) para cada planta do split indicado.

    Parâmetros:
        plantas - lista de nomes (ex: ['Tomato', 'Potato']); None para todas
        n       - pares por planta (pega n doenças diferentes)
        caminho - raiz do dataset
        split   - 'train' ou 'val'
        semente - semente aleatória

    Retorna lista de dicts:
        {'planta', 'classe_saudavel', 'classe_doente', 'img_saudavel', 'img_doente'}
    """
    random.seed(semente)
    pasta = _pasta_split(caminho or PASTA_DATASET, split)
    classes = listar_classes(caminho, split=split)
    saudaveis, doentes_lista = separar_saudaveis_doentes(classes)

    mapa_saudavel = {_planta_de_classe(c): c for c in saudaveis}
    mapa_doente: dict = {}
    for c in doentes_lista:
        mapa_doente.setdefault(_planta_de_classe(c), []).append(c)

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
            pares.append({
                'planta'         : planta,
                'classe_saudavel': cls_s,
                'classe_doente'  : cls_d,
                'img_saudavel'   : carregar_imagem(random.choice(imgs_s)),
                'img_doente'     : carregar_imagem(random.choice(imgs_d)),
            })

    return pares


# ─── Helpers de nomenclatura ─────────────────────────────────────────────────

def separar_saudaveis_doentes(classes):
    """Divide lista de classes em (saudaveis, doentes)."""
    saudaveis = [c for c in classes if 'healthy' in c.lower()]
    doentes   = [c for c in classes if 'healthy' not in c.lower()]
    return saudaveis, doentes


def _planta_de_classe(nome_classe):
    """'Tomato___Early_blight' → 'Tomato'"""
    for sep in ['___', '__', '_']:
        if sep in nome_classe:
            return nome_classe.split(sep)[0]
    return nome_classe


def nome_legivel(nome_classe):
    """'Tomato___Early_blight' → 'Tomato - Early Blight'"""
    return nome_classe.replace('___', ' - ').replace('__', ' ').replace('_', ' ').title()

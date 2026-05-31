"""
API do dataset Plant Disease Detection (Roboflow / YOLOv8)
"""
import os
import glob
import random
import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


def nome_legivel(cls):
    return str(cls).replace('_', ' ').replace('-', ' ').strip().title()


def caminho_data_yaml(data_dir):
    return os.path.join(data_dir, 'data.yaml')


def carregar_data_yaml(data_dir):
    with open(caminho_data_yaml(data_dir), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def verificar_dataset(data_dir):
    return (os.path.exists(caminho_data_yaml(data_dir))
            and os.path.isdir(os.path.join(data_dir, 'train', 'images')))


def listar_classes(data_dir):
    cfg = carregar_data_yaml(data_dir)
    nomes = cfg.get('names', [])
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes, key=lambda x: int(x))]
    return list(nomes)


def listar_imagens(data_dir, split='train'):
    pasta = os.path.join(data_dir, split, 'images')
    arquivos = []
    for e in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG', '*.JPEG'):
        arquivos += glob.glob(os.path.join(pasta, e))
    return sorted(set(arquivos))


def caminho_label(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    pasta_labels = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels')
    return os.path.join(pasta_labels, base + '.txt')


def ler_labels_yolo(img_path, shape):
    """Le labels YOLO (classe xc yc w h normalizados) -> (classe_id, x1, y1, x2, y2) pixels."""
    h, w = shape[:2]
    lp = caminho_label(img_path)
    caixas = []
    if not os.path.exists(lp):
        return caixas
    with open(lp, 'r', encoding='utf-8') as f:
        for linha in f:
            p = linha.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            xc, yc, bw, bh = map(float, p[1:5])
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))
            caixas.append((cid, x1, y1, x2, y2))
    return caixas


def carregar_imagem_rgb(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def amostrar_imagens(data_dir, n=8, split='train', com_label=True, semente=0):
    """Amostra n imagens; se com_label, prioriza imagens com caixas anotadas."""
    random.seed(semente)
    arquivos = listar_imagens(data_dir, split)
    random.shuffle(arquivos)
    sel = []
    for p in arquivos:
        if com_label:
            lp = caminho_label(p)
            if not (os.path.exists(lp) and os.path.getsize(lp) > 0):
                continue
        sel.append(p)
        if len(sel) >= n:
            break
    if not sel:
        sel = arquivos[:n]
    return sel


def amostrar_por_classe(data_dir, classes, n_por_classe=1, split='train', semente=0):
    """Retorna ate n_por_classe caminhos para cada classe, garantindo diversidade."""
    random.seed(semente)
    arquivos = listar_imagens(data_dir, split)
    random.shuffle(arquivos)
    resultado = {i: [] for i in range(len(classes))}
    for p in arquivos:
        lp = caminho_label(p)
        if not (os.path.exists(lp) and os.path.getsize(lp) > 0):
            continue
        try:
            with open(lp, 'r', encoding='utf-8') as f:
                ids = {int(float(l.split()[0])) for l in f if len(l.split()) >= 5}
        except Exception:
            continue
        for cid in ids:
            if cid in resultado and len(resultado[cid]) < n_por_classe:
                resultado[cid].append(p)
        if all(len(v) >= n_por_classe for v in resultado.values()):
            break
    return resultado


def desenhar_boxes(img_rgb, caixas, classes=None, cor=(0, 160, 255), espessura=2):
    out = img_rgb.copy()
    for c in caixas:
        cid, x1, y1, x2, y2 = c
        cv2.rectangle(out, (x1, y1), (x2, y2), cor, espessura)
        rot = nome_legivel(classes[cid]) if (classes and 0 <= cid < len(classes)) else str(cid)
        cv2.putText(out, rot, (x1 + 2, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
    return out


def estatisticas_dataset(data_dir):
    s = {'classes': listar_classes(data_dir)}
    s['n_classes'] = len(s['classes'])
    for split in ('train', 'valid', 'test'):
        s[split] = len(listar_imagens(data_dir, split))
    return s


def preparar_yaml_treino(data_dir):
    """Reescreve data.yaml com caminhos absolutos para o Ultralytics treinar sem erro de path."""
    nomes = listar_classes(data_dir)
    novo = {
        'path':  data_dir,
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    len(nomes),
        'names': nomes,
    }
    destino = os.path.join(data_dir, 'data_treino.yaml')
    with open(destino, 'w', encoding='utf-8') as f:
        yaml.safe_dump(novo, f, sort_keys=False, allow_unicode=True)
    return destino


def score_imagem(img_path, criterio, tamanho=(416, 416)):
    """
    Pontua uma imagem por um criterio visual.
    criterio: 'ruido' (Laplacian var), 'contraste' (bimodalidade), 'bordas' (Canny px)
    """
    try:
        bgr = cv2.resize(cv2.imread(img_path), tamanho)
        if bgr is None:
            return 0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if criterio == 'ruido':
            return float(cv2.Laplacian(g, cv2.CV_64F).var())
        if criterio == 'contraste':
            h = np.histogram(g, bins=32)[0].astype(float)
            h /= h.sum() + 1e-9
            return float(h[:16].sum() * h[16:].sum())
        if criterio == 'bordas':
            b = cv2.GaussianBlur(g, (5, 5), 0)
            return float(cv2.Canny(b, 50, 150).sum())
    except Exception:
        return 0

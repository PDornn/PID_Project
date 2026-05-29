"""
Deteccao de doencas em plantas com YOLOv8 (DETECCAO DE OBJETOS).
Dataset: Plant Disease Detection (Roboflow, formato YOLOv8 com bounding boxes).

Este modulo:
  1. Carrega/treina o YOLOv8 para LOCALIZAR regioes de doenca (bounding boxes)
  2. Roda inferencia em imagens novas
  3. Fornece metricas (IoU / recall / precision) para comparar o YOLO
     com o metodo classico HSV e com o ground-truth do dataset

Referencia: https://docs.ultralytics.com/
"""

import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
    YOLO_DISPONIVEL = True
except ImportError:
    YOLO_DISPONIVEL = False
    print("[YOLO] Ultralytics nao instalado. Execute: pip install ultralytics")

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_MODELS = os.path.join(PROJETO_RAIZ, 'models')
PASTA_RESULTS = os.path.join(PROJETO_RAIZ, 'results', 'yolo')


def _checar_yolo():
    if not YOLO_DISPONIVEL:
        raise ImportError("Instale o ultralytics: pip install ultralytics")


def _caminho_modelo(nome):
    """Retorna o caminho local do modelo; se nao existir, usa o nome direto
    (Ultralytics faz o download automatico do checkpoint pre-treinado)."""
    os.makedirs(PASTA_MODELS, exist_ok=True)
    local = os.path.join(PASTA_MODELS, nome)
    return local if os.path.exists(local) else nome


# ─── Carregamento de modelos ─────────────────────────────────────────────────

def carregar_modelo_deteccao(nome='yolov8n.pt'):
    """
    Carrega YOLOv8 para deteccao de objetos.

    Modelos disponiveis (do menor/mais rapido ao maior/mais preciso):
      yolov8n.pt  yolov8s.pt  yolov8m.pt  yolov8l.pt  yolov8x.pt
    """
    _checar_yolo()
    return YOLO(_caminho_modelo(nome))


def carregar_modelo_custom(caminho_weights):
    """Carrega qualquer modelo YOLO a partir de um arquivo .pt local (ex: best.pt)."""
    _checar_yolo()
    if not os.path.exists(caminho_weights):
        raise FileNotFoundError(f"Arquivo de pesos nao encontrado: {caminho_weights}")
    return YOLO(caminho_weights)


# ─── Treinamento (deteccao) ──────────────────────────────────────────────────

def treinar_detector(
    data_yaml,
    epocas=30,
    imgsz=640,
    batch=16,
    modelo_base='yolov8n.pt',
    nome_experimento='plant_disease',
    project=None,
    device=None,
):
    """
    Fine-tuning do YOLOv8 para DETECCAO de doencas no dataset Roboflow (formato YOLOv8).

    O 'data_yaml' deve apontar para um arquivo data.yaml valido com as chaves:
      path, train, val, test, nc, names

    Parametros:
        data_yaml         - caminho do arquivo data.yaml
        epocas            - epocas de treinamento (30 e um bom ponto de partida)
        imgsz             - tamanho das imagens de treino (640 padrao do YOLOv8)
        batch             - tamanho do batch
        modelo_base       - checkpoint inicial (yolov8n.pt = nano, rapido)
        nome_experimento  - nome da pasta de saida
        project           - diretorio raiz das runs (default: data/models/runs)

    Retorna:
        (modelo_treinado, resultado)  onde resultado.save_dir aponta para a run.
    """
    _checar_yolo()
    if project is None:
        project = os.path.join(PROJETO_RAIZ, 'models', 'runs')

    modelo = YOLO(modelo_base)
    resultado = modelo.train(
        data=data_yaml,
        task='detect',
        epochs=epocas,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=nome_experimento,
        exist_ok=True,
        verbose=False,
        device=device,
    )
    return modelo, resultado


def avaliar_modelo(modelo, data_yaml, split='test'):
    """Roda a validacao do Ultralytics e retorna as metricas (mAP50, mAP50-95, etc)."""
    _checar_yolo()
    metrics = modelo.val(data=data_yaml, split=split, verbose=False)
    return {
        'map50': float(getattr(metrics.box, 'map50', 0.0)),
        'map50_95': float(getattr(metrics.box, 'map', 0.0)),
        'precision': float(np.mean(metrics.box.p)) if len(metrics.box.p) else 0.0,
        'recall': float(np.mean(metrics.box.r)) if len(metrics.box.r) else 0.0,
    }


# ─── Inferencia ──────────────────────────────────────────────────────────────

def detectar_objetos(img_rgb, modelo=None, confianca=0.25, iou=0.45):
    """
    Roda inferencia de deteccao (bounding boxes) em uma imagem RGB.

    Parametros:
        img_rgb   - imagem RGB (numpy uint8)
        modelo    - instancia YOLO; None carrega yolov8n.pt
        confianca - limiar minimo de confianca
        iou       - limiar de IoU para o NMS

    Retorna:
        img_anotada - imagem RGB com bboxes e rotulos desenhados
        deteccoes   - lista de dicts: {'classe', 'confianca', 'bbox': (x1,y1,x2,y2)}
    """
    _checar_yolo()
    if modelo is None:
        modelo = carregar_modelo_deteccao()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    resultados = modelo(img_bgr, conf=confianca, iou=iou, verbose=False)

    img_anotada = img_rgb.copy()
    deteccoes = []

    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_nome = modelo.names[int(box.cls[0])]

            cv2.rectangle(img_anotada, (x1, y1), (x2, y2), (0, 200, 50), 2)
            rotulo = f'{cls_nome} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_anotada, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 200, 50), -1)
            cv2.putText(img_anotada, rotulo, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            deteccoes.append({'classe': cls_nome, 'confianca': conf, 'bbox': (x1, y1, x2, y2)})

    return img_anotada, deteccoes


# ─── Metricas de comparacao (IoU) ────────────────────────────────────────────

def iou_xyxy(a, b):
    """IoU entre duas caixas no formato (x1, y1, x2, y2)."""
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    uniao = areaA + areaB - inter
    return inter / uniao if uniao > 0 else 0.0


def casar_caixas(gt, pred, iou_min=0.3):
    """
    Compara caixas previstas (pred) com o ground-truth (gt), ambas em (x1,y1,x2,y2).

    Faz um matching guloso: cada caixa GT busca a melhor caixa prevista ainda livre.
    Um acerto (TP) ocorre quando IoU >= iou_min.

    Retorna dict com: n_gt, n_pred, tp, recall, precision, iou_medio.
    """
    usados = set()
    tp = 0
    ious = []
    for g in gt:
        melhor = 0.0
        melhor_j = -1
        for j, p in enumerate(pred):
            if j in usados:
                continue
            v = iou_xyxy(g, p)
            if v > melhor:
                melhor = v
                melhor_j = j
        if melhor >= iou_min and melhor_j >= 0:
            usados.add(melhor_j)
            tp += 1
            ious.append(melhor)

    n_gt = len(gt)
    n_pred = len(pred)
    return {
        'n_gt': n_gt,
        'n_pred': n_pred,
        'tp': tp,
        'recall': tp / n_gt if n_gt else 0.0,
        'precision': tp / n_pred if n_pred else 0.0,
        'iou_medio': float(np.mean(ious)) if ious else 0.0,
    }


# ─── Utilitarios visuais ─────────────────────────────────────────────────────

def comparar_metodos(img_rgb, resultado_classico, resultado_yolo,
                     label_classico='Classico', label_yolo='YOLO'):
    """
    Gera uma imagem lado a lado: Original | metodo classico | YOLO.
    Todas as imagens de entrada sao RGB.
    """
    h = max(img_rgb.shape[0], resultado_classico.shape[0], resultado_yolo.shape[0])
    alvo = (320, h)

    def _resize(im):
        return cv2.resize(im, alvo) if im.shape[:2][::-1] != alvo else im.copy()

    orig = _resize(img_rgb)
    clas = _resize(resultado_classico if len(resultado_classico.shape) == 3
                   else cv2.cvtColor(resultado_classico, cv2.COLOR_GRAY2RGB))
    yolo = _resize(resultado_yolo)

    def _rotular(im, label):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 26), (30, 30, 30), -1)
        cv2.putText(im, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return im

    paineis = [
        _rotular(orig, 'Original'),
        _rotular(clas, label_classico),
        _rotular(yolo, label_yolo),
    ]
    return np.hstack(paineis)


def salvar_resultado_yolo(img_rgb, nome_arquivo):
    """Salva uma imagem anotada na pasta results/yolo/."""
    os.makedirs(PASTA_RESULTS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTS, nome_arquivo)
    cv2.imwrite(caminho, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f'[YOLO] Salvo: {caminho}')
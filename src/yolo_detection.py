"""
Detecção com YOLO - Capstone de Processamento Digital de Imagens
Dataset: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Usa YOLOv8 (Ultralytics) para:
  1. Detecção geral de objetos com modelo pré-treinado COCO (yolov8n.pt)
  2. Classificação de doenças em folhas com YOLOv8-cls fine-tuning

Referência: https://docs.ultralytics.com/
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
PASTA_MODELS = os.path.join(PROJETO_RAIZ, 'data', 'models')
PASTA_RESULTS = os.path.join(PROJETO_RAIZ, 'results', 'yolo')


def _checar_yolo():
    if not YOLO_DISPONIVEL:
        raise ImportError("Instale o ultralytics: pip install ultralytics")


def _caminho_modelo(nome):
    """Retorna caminho local do modelo; se não existir, usa nome direto (download automático)."""
    os.makedirs(PASTA_MODELS, exist_ok=True)
    local = os.path.join(PASTA_MODELS, nome)
    return local if os.path.exists(local) else nome


# ─── Carregamento de modelos ─────────────────────────────────────────────────

def carregar_modelo_deteccao(nome='yolov8n.pt'):
    """
    Carrega YOLOv8 para detecção de objetos (pré-treinado COCO).

    Modelos disponíveis (do menor/mais rápido ao maior/mais preciso):
      yolov8n.pt  yolov8s.pt  yolov8m.pt  yolov8l.pt  yolov8x.pt
    """
    _checar_yolo()
    return YOLO(_caminho_modelo(nome))


def carregar_modelo_classificacao(nome='yolov8n-cls.pt'):
    """
    Carrega YOLOv8 no modo classificação (ImageNet pré-treinado).
    Pode ser substituído por um modelo fine-tuned no PlantVillage.
    """
    _checar_yolo()
    return YOLO(_caminho_modelo(nome))


def carregar_modelo_custom(caminho_weights):
    """Carrega qualquer modelo YOLO a partir de um arquivo .pt local."""
    _checar_yolo()
    if not os.path.exists(caminho_weights):
        raise FileNotFoundError(f"Arquivo de pesos nao encontrado: {caminho_weights}")
    return YOLO(caminho_weights)


# ─── Inferência ──────────────────────────────────────────────────────────────

def detectar_objetos(img_rgb, modelo=None, confianca=0.25, iou=0.45):
    """
    Roda inferência de detecção de objetos (bounding boxes) em uma imagem.

    Parâmetros:
        img_rgb   - imagem RGB (numpy uint8)
        modelo    - instância YOLO; None carrega yolov8n.pt
        confianca - limiar mínimo de confiança
        iou       - limiar de IoU para NMS

    Retorna:
        img_anotada - imagem RGB com bboxes e rótulos desenhados
        deteccoes   - lista de dicts: {'classe', 'confianca', 'bbox'}
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
            (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img_anotada, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 50), -1)
            cv2.putText(img_anotada, rotulo, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            deteccoes.append({'classe': cls_nome, 'confianca': conf, 'bbox': (x1, y1, x2, y2)})

    return img_anotada, deteccoes


def classificar_folha(img_rgb, modelo_cls=None, top_k=5):
    """
    Classifica a imagem de folha usando YOLOv8-cls.

    Com modelo ImageNet pré-treinado mostra as top-k classes gerais.
    Com modelo fine-tuned no PlantVillage mostra doenças detectadas.

    Parâmetros:
        img_rgb   - imagem RGB da folha
        modelo_cls - instância YOLO-cls; None carrega yolov8n-cls.pt
        top_k     - número de melhores predições a retornar

    Retorna:
        img_anotada - imagem RGB com rótulo da classe mais provável
        predicoes   - lista de dicts: {'classe', 'confianca'}
    """
    _checar_yolo()
    if modelo_cls is None:
        modelo_cls = carregar_modelo_classificacao()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    resultados = modelo_cls(img_bgr, verbose=False)

    predicoes = []
    img_anotada = img_rgb.copy()

    for r in resultados:
        probs = r.probs
        if probs is None:
            continue
        indices = probs.top5[:top_k]
        for idx in indices:
            predicoes.append({
                'classe': modelo_cls.names[int(idx)],
                'confianca': float(probs.data[int(idx)]),
            })

    if predicoes:
        melhor = predicoes[0]
        texto = f"{melhor['classe']}: {melhor['confianca']:.1%}"
        h = img_anotada.shape[0]
        cv2.rectangle(img_anotada, (0, h - 40), (img_anotada.shape[1], h), (0, 0, 0), -1)
        cv2.putText(img_anotada, texto, (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return img_anotada, predicoes


def classificar_lote(amostras, modelo_cls=None, top_k=3):
    """
    Classifica uma lista de amostras (dicts com chave 'img' e 'classe').

    Retorna lista de dicts com campos originais + 'predicao_yolo' e 'confianca_yolo'.
    """
    _checar_yolo()
    if modelo_cls is None:
        modelo_cls = carregar_modelo_classificacao()

    resultados = []
    for amostra in amostras:
        img_anotada, preds = classificar_folha(amostra['img'], modelo_cls, top_k)
        item = dict(amostra)
        item['img_yolo'] = img_anotada
        item['predicao_yolo'] = preds[0]['classe'] if preds else 'N/A'
        item['confianca_yolo'] = preds[0]['confianca'] if preds else 0.0
        item['top_k_preds'] = preds
        resultados.append(item)

    return resultados


# ─── Treinamento ─────────────────────────────────────────────────────────────

def treinar_classificador(
    caminho_dataset,
    epocas=10,
    imgsz=224,
    modelo_base='yolov8n-cls.pt',
    nome_experimento='plantvillage_cls',
):
    """
    Fine-tuning de YOLOv8-cls no dataset PlantVillage.

    O dataset deve ter subpastas por classe:
      data/PlantVillage/
        ├── Tomato___Early_blight/
        ├── Tomato___healthy/
        └── ...

    Parâmetros:
        caminho_dataset    - raiz do dataset com subpastas por classe
        epocas             - épocas de treinamento (10 é suficiente para demo)
        imgsz              - tamanho das imagens (224 padrão)
        modelo_base        - checkpoint inicial
        nome_experimento   - nome da pasta de saída em results/yolo_train/

    Retorna o modelo treinado.
    """
    _checar_yolo()
    os.makedirs(PASTA_RESULTS, exist_ok=True)

    modelo = YOLO(modelo_base)
    modelo.train(
        data=caminho_dataset,
        epochs=epocas,
        imgsz=imgsz,
        task='classify',
        project=os.path.join(PROJETO_RAIZ, 'results', 'yolo_train'),
        name=nome_experimento,
        exist_ok=True,
    )
    return modelo


# ─── Utilitários ─────────────────────────────────────────────────────────────

def comparar_metodos(img_rgb, resultado_classico, resultado_yolo, label_classico='Clássico', label_yolo='YOLO'):
    """
    Gera uma imagem lado a lado comparando detecção clássica com YOLO.

    Parâmetros:
        img_rgb          - imagem original RGB
        resultado_classico - imagem RGB com anotações do método clássico
        resultado_yolo   - imagem RGB com anotações do YOLO
        label_classico   - rótulo do painel esquerdo
        label_yolo       - rótulo do painel direito

    Retorna imagem combinada (numpy RGB).
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
    """Salva imagem anotada pelo YOLO na pasta results/yolo/."""
    os.makedirs(PASTA_RESULTS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTS, nome_arquivo)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(caminho, img_bgr)
    print(f'[YOLO] Salvo: {caminho}')

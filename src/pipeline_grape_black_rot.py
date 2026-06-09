"""
Pipeline clássico completo para detecção de lesões — Grape Black Rot (Amostra A)

Etapas:
  1. Gaussian Blur (5×5)         → redução de ruído
  2. Máscara folha (HSV verde)   → isola a folha do fundo
  3. Máscara lesão (HSV escuro)  → dentro da folha, detecta manchas
  4. Morfologia (OPEN → CLOSE)   → limpa ruído e fecha lesões parciais
  5. Componentes conectados      → extrai e filtra lesões por área
  6. Canny por ROI               → refina/caracteriza a borda de cada lesão

Executar:
    python src/pipeline_grape_black_rot.py
"""

import sys
import os
from pathlib import Path

# ── Raiz do projeto ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sem janela interativa — salva o PNG direto
import matplotlib.pyplot as plt

from dataset import (
    amostrar_imagens, ler_labels_yolo, carregar_imagem_rgb,
    listar_classes, nome_legivel,
)
from yolo_detection import casar_caixas

# ── Constantes ─────────────────────────────────────────────────────────────────
DATA_DIR  = str(ROOT / "data")
TAMANHO   = (416, 416)
SAIDA_DIR = str(ROOT / "results" / "07_pipeline_grape")
os.makedirs(SAIDA_DIR, exist_ok=True)

CLASSES = listar_classes(DATA_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPAS DO PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def etapa1_blur(img_rgb):
    """Gaussian Blur 5×5 — reduz ruído preservando cor."""
    bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    suave = cv2.GaussianBlur(bgr, (5, 5), 0)
    return cv2.cvtColor(suave, cv2.COLOR_BGR2RGB)


def etapa2_mascara_folha(img_rgb):
    """HSV verde + Canny na máscara HSV → fill do contorno → fecha lesões.

    1. Máscara HSV verde isola os pixels de folha pelo tom.
    2. Canny aplicado sobre a máscara HSV (binária, limpa) encontra
       bordas precisas sem interferência do fundo — contorno sempre fechado.
    3. fillPoly do maior contorno define o limite externo rígido.
    4. AND com a máscara HSV garante que só pixels verdes passem —
       elimina qualquer fundo que o fill possa ter incluído.
    5. Morfologia CLOSE fecha os buracos das lesões escuras no interior.
    6. Maior componente conectado remove pixels verdes espúrios do fundo.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 1. Máscara HSV verde
    verde = cv2.inRange(hsv, np.array([25, 40, 30]), np.array([90, 255, 255]))

    # 2. Canny na máscara verde (imagem binária — bordas sempre fechadas)
    blur_verde = cv2.GaussianBlur(verde, (7, 7), 0)
    bordas = cv2.Canny(blur_verde, 30, 90)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bordas_d = cv2.dilate(bordas, k5, iterations=2)

    # 3. Fill do maior contorno → limite rígido da borda
    contornos, _ = cv2.findContours(bordas_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    limite = np.zeros(verde.shape, dtype=np.uint8)
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        cv2.fillPoly(limite, [maior], 255)

    # 4. AND com HSV verde — remove fundo incluído pelo fill
    mascara = cv2.bitwise_and(limite, verde)

    # 5. Fecha buracos das lesões escuras dentro da folha
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k_close, iterations=3)
    mascara = cv2.bitwise_and(mascara, limite)

    # 6. Mantém só o maior componente (exclui pixels verdes espúrios do fundo)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n > 1:
        maior_idx = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]))
        mascara = ((labels == maior_idx) * 255).astype(np.uint8)

    return mascara


def etapa3_mascara_lesao(img_rgb, mascara_folha):
    """
    HSV marrom/escuro — detecta lesões dentro da folha.

    Cor de referência medida nas manchas: R=35 G=16 B=7 → H≈10, S≈200, V≈35
    Faixas limitadas a V<130 para não capturar tecido verde-amarelado da folha.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    faixas = [
        # quase-preto / escleródio — qualquer matiz, S baixo, V ≤ 45
        (np.array([0,   0,  0]),  np.array([179, 100,  45])),
        # centro escuro/preto — tom marrom, V até 90
        (np.array([0,  20,  0]),  np.array([ 20, 255,  90])),
        # halo marrom escuro (H<20 exclui amarelo-verde)
        (np.array([0,  40, 30]),  np.array([ 20, 255, 160])),
        # borda marrom-laranja escura — V<110 impede capturar folha
        (np.array([20, 80, 30]),  np.array([ 35, 255, 110])),
    ]

    combinada = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in faixas:
        combinada = cv2.bitwise_or(combinada, cv2.inRange(hsv, lo, hi))

    return combinada


def etapa4_morfologia(mascara_raw, mascara_folha):
    """
    OPEN remove speckle, CLOSE fecha buracos dentro das lesões.
    Aplica a máscara da folha para eliminar falsos positivos fora dela.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mascara_raw, cv2.MORPH_OPEN,  kernel, iterations=1)
    m = cv2.morphologyEx(m,           cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.bitwise_and(m, mascara_folha)
    return m


def etapa5_componentes(mascara, area_min=80, area_max=None):
    """
    Componentes conectados + filtro de área.
    Retorna máscara final, lista de (x, y, w, h) e stats de cada lesão.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara)

    mascara_final = np.zeros_like(mascara)
    bboxes, areas = [], []

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min:
            continue
        if area_max is not None and area > area_max:
            continue
        mascara_final[labels == i] = 255
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, w, h))
        areas.append(int(area))

    return mascara_final, bboxes, areas


def etapa6_canny_por_roi(img_rgb, bboxes, margem=4):
    """
    Canny aplicado dentro de cada ROI de lesão.
    Retorna a imagem original com as bordas de cada lesão destacadas em ciano.
    """
    resultado = img_rgb.copy()
    h_img, w_img = img_rgb.shape[:2]

    for (x, y, w, h) in bboxes:
        x0 = max(0, x - margem)
        y0 = max(0, y - margem)
        x1 = min(w_img, x + w + margem)
        y1 = min(h_img, y + h + margem)

        roi_rgb = img_rgb[y0:y1, x0:x1]
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)

        med = float(np.median(roi_blur))
        t1  = int(max(0,   0.66 * med))
        t2  = int(min(255, 1.33 * med))
        bordas = cv2.Canny(roi_blur, t1, t2)

        # Desenha bordas ciano sobre a imagem resultado
        resultado[y0:y1, x0:x1][bordas > 0] = [0, 220, 220]

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def _mostrar(ax, img, titulo, cmap=None):
    ax.imshow(img, cmap=cmap)
    ax.set_title(titulo, fontsize=9, pad=4)
    ax.axis("off")


def gerar_figura_pipeline(img_orig, img_blur, mascara_folha, mascara_raw,
                           mascara_morfo, mascara_final, img_canny_roi,
                           img_diagnostico, bboxes, areas, gt_boxes,
                           nome_classe):
    """Salva grade com todas as etapas do pipeline."""

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    axs = axs.flatten()

    # Linha 1
    _mostrar(axs[0], img_orig,                                    "Original")
    _mostrar(axs[1], img_blur,                                    "Etapa 1 — Gaussian Blur (5×5)")
    _mostrar(axs[2], mascara_folha,   "Etapa 2 — Máscara Folha (HSV verde)",  cmap="gray")
    _mostrar(axs[3], mascara_raw,     "Etapa 3 — Máscara Lesão (HSV bruto)",  cmap="gray")

    # Linha 2
    _mostrar(axs[4], mascara_morfo,   "Etapa 4 — Morfologia (OPEN→CLOSE)",    cmap="gray")
    _mostrar(axs[5], mascara_final,   "Etapa 5 — Componentes (área filtrada)", cmap="gray")
    _mostrar(axs[6], img_canny_roi,   "Etapa 6 — Canny por ROI (bordas ciano)")
    _mostrar(axs[7], img_diagnostico, "Resultado Final — Diagnóstico")

    titulo = (f"Pipeline Grape Black Rot — Amostra A\n"
              f"{len(bboxes)} lesões detectadas | GT: {len(gt_boxes)} caixas | "
              f"Áreas: {', '.join(str(a) for a in sorted(areas, reverse=True)[:5])} px")
    fig.suptitle(titulo, fontsize=11, y=1.01)
    plt.tight_layout()

    caminho = os.path.join(SAIDA_DIR, "pipeline_amostra_A.png")
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return caminho


def gerar_figura_comparacao(img_orig, img_diagnostico, gt_boxes, hsv_bboxes,
                             hsv_recall, hsv_precision, hsv_iou, nome_classe):
    """Salva comparação GT vs resultado do pipeline."""
    img_gt = img_orig.copy()
    for (_, x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 160, 255), 2)
        cv2.putText(img_gt, "GT", (x1 + 2, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)

    img_hsv = img_orig.copy()
    for (x, y, w, h) in hsv_bboxes:
        cv2.rectangle(img_hsv, (x, y), (x + w, y + h), (255, 200, 0), 2)
    cv2.putText(img_hsv,
                f"HSV | R={hsv_recall*100:.0f}% P={hsv_precision*100:.0f}% IoU={hsv_iou:.2f}",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    _mostrar(axs[0], img_orig,        "Original")
    _mostrar(axs[1], img_gt,          f"Ground-Truth ({len(gt_boxes)} caixas)")
    _mostrar(axs[2], img_diagnostico, f"Pipeline HSV ({len(hsv_bboxes)} lesões)")

    plt.suptitle(f"Comparação — {nome_classe}", fontsize=11)
    plt.tight_layout()

    caminho = os.path.join(SAIDA_DIR, "comparacao_gt_vs_pipeline.png")
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO VISUAL FINAL
# ══════════════════════════════════════════════════════════════════════════════

def montar_diagnostico(img_rgb, mascara_final, bboxes):
    """Blend vermelho nas lesões + bboxes amarelas + texto."""
    # Blend
    destaque   = np.zeros_like(img_rgb)
    destaque[:] = (220, 50, 50)
    m3 = cv2.merge([mascara_final, mascara_final, mascara_final]).astype(np.float32) / 255.0
    resultado = (img_rgb.astype(np.float32) * (1 - 0.5 * m3)
                 + destaque.astype(np.float32) * (0.5 * m3))
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)

    # Bboxes
    for (x, y, w, h) in bboxes:
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (255, 220, 0), 2)

    # Rodapé
    h_img = resultado.shape[0]
    n = len(bboxes)
    texto = f"Grape Black Rot  |  {n} lesao{'es' if n != 1 else ''} detectada{'s' if n != 1 else ''}"
    cv2.rectangle(resultado, (0, h_img - 30), (resultado.shape[1], h_img), (20, 20, 20), -1)
    cv2.putText(resultado, texto, (8, h_img - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Pipeline Grape Black Rot — Amostra A")
    print("=" * 60)

    # ── Carrega a mesma amostra A do notebook (seed=7) ─────────────────────
    sel    = amostrar_imagens(DATA_DIR, n=12, split="train", com_label=True, semente=7)
    path_a = sel[0]
    path_b = sel[1]

    img_a  = cv2.resize(carregar_imagem_rgb(path_a), TAMANHO)
    img_b  = cv2.resize(carregar_imagem_rgb(path_b), TAMANHO)
    gt_a   = ler_labels_yolo(path_a, img_a.shape)
    gt_b   = ler_labels_yolo(path_b, img_b.shape)

    # A fica com mais caixas (igual ao notebook)
    if len(gt_b) > len(gt_a):
        img_a, gt_a, path_a = img_b, gt_b, path_b

    nome_classe = ", ".join(sorted({nome_legivel(CLASSES[c[0]]) for c in gt_a})) or "?"
    print(f"\nAmostra A : {os.path.basename(path_a)}")
    print(f"Classe    : {nome_classe}")
    print(f"GT boxes  : {len(gt_a)}")

    # ── Pipeline ───────────────────────────────────────────────────────────
    print("\n[1/6] Gaussian Blur 5×5 ...")
    img_blur = etapa1_blur(img_a)

    print("[2/6] Máscara folha (HSV verde) ...")
    mascara_folha = etapa2_mascara_folha(img_blur)

    print("[3/6] Máscara lesão (HSV escuro/marrom) ...")
    mascara_raw = etapa3_mascara_lesao(img_blur, mascara_folha)

    print("[4/6] Morfologia OPEN → CLOSE ...")
    mascara_morfo = etapa4_morfologia(mascara_raw, mascara_folha)

    print("[5/6] Componentes conectados + filtro de área ...")
    mascara_final, bboxes, areas = etapa5_componentes(mascara_morfo, area_min=80)

    print("[6/6] Canny por ROI ...")
    img_canny_roi = etapa6_canny_por_roi(img_a, bboxes)

    # ── Diagnóstico visual ─────────────────────────────────────────────────
    img_diagnostico = montar_diagnostico(img_a, mascara_final, bboxes)

    # ── Métricas vs Ground-Truth ───────────────────────────────────────────
    hsv_xyxy = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
    gt_xyxy  = [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in gt_a]
    metricas = casar_caixas(gt_xyxy, hsv_xyxy, iou_min=0.2)

    # ── Relatório ──────────────────────────────────────────────────────────
    area_folha  = int(mascara_folha.sum()) // 255
    area_lesao  = int(mascara_final.sum()) // 255
    total_px    = TAMANHO[0] * TAMANHO[1]

    print()
    print("-" * 40)
    print("  RESULTADOS DO PIPELINE")
    print("-" * 40)
    print(f"  Área folha           : {area_folha:,} px ({area_folha/total_px*100:.1f}%)")
    print(f"  Área lesionada (px)  : {area_lesao:,} px ({area_lesao/area_folha*100:.1f}% da folha)")
    print(f"  Lesões detectadas    : {len(bboxes)}")
    if areas:
        print(f"  Área por lesão (px)  : min={min(areas)}  max={max(areas)}  med={int(np.median(areas))}")
    print()
    print(f"  GT boxes             : {metricas['n_gt']}")
    print(f"  TP (IoU ≥ 0.2)       : {metricas['tp']}")
    print(f"  Recall               : {metricas['recall']*100:.1f}%")
    print(f"  Precision            : {metricas['precision']*100:.1f}%")
    print(f"  IoU médio            : {metricas['iou_medio']:.2f}")
    print()

    # ── Salva figuras ──────────────────────────────────────────────────────
    p1 = gerar_figura_pipeline(
        img_a, img_blur, mascara_folha, mascara_raw,
        mascara_morfo, mascara_final, img_canny_roi,
        img_diagnostico, bboxes, areas, gt_a, nome_classe,
    )
    p2 = gerar_figura_comparacao(
        img_a, img_diagnostico, gt_a, bboxes,
        metricas["recall"], metricas["precision"], metricas["iou_medio"],
        nome_classe,
    )

    print(f"  Figura pipeline   → {p1}")
    print(f"  Figura comparação → {p2}")
    print()
    print("Concluído.")


if __name__ == "__main__":
    main()

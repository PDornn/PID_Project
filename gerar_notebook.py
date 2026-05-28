"""
Gera o notebook principal do Capstone de PDI.
Execute: .venv/Scripts/python gerar_notebook.py
"""
import nbformat as nbf
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(PROJECT_ROOT, 'notebook', 'capstone_plant_disease.ipynb')

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python (capstone-pdi)",
        "language": "python",
        "name": "capstone-pdi",
    },
    "language_info": {"name": "python", "version": "3.13.0"},
})

def M(text): return nbf.v4.new_markdown_cell(text.strip())
def C(code): return nbf.v4.new_code_cell(code.strip())

cells = [

# ── Título ────────────────────────────────────────────────────────────────────
M("""
# Capstone Project — Detecção de Doenças em Plantas

**Processamento Digital de Imagens · Ciência da Computação — 8º Período**
Centro Universitário Dom Helder · Prof. Dr. Fischer Stefan
Dataset: [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

---

| Bloco | Técnica |
|---|---|
| 1 | **Threshold & Blur** — suavização e limiarização |
| 2 | **Edge & Corner Detection** — Canny, Sobel, Harris, Shi-Tomasi |
| 3 | **Object Detection + ROI + Blend/Paste** — segmentação HSV + composição visual |
| 4 | **YOLO** — YOLOv8-cls para classificação de doenças |

> Execute **Run All** para rodar o pipeline completo. Todos os resultados são salvos em `results/`.
"""),

# ── Setup ─────────────────────────────────────────────────────────────────────
C("""
%matplotlib inline
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join('..', 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from import_dataset import (
    verificar_dataset, listar_classes, estatisticas_dataset,
    amostrar_por_classe, amostrar_pares, separar_saudaveis_doentes,
    nome_legivel, PASTA_DATASET,
)
from preprocessing import (
    converter_grayscale, aplicar_blur_gaussiano, aplicar_blur_mediano,
    aplicar_blur_bilateral, aplicar_threshold_simples,
    aplicar_threshold_otsu, aplicar_threshold_adaptativo_gaussiano,
)
from detection import (
    detectar_bordas_canny, detectar_bordas_sobel,
    detectar_cantos_harris, detectar_cantos_shi_tomasi,
    detectar_contornos_externos,
    segmentar_folha_verde, detectar_lesoes_hsv,
    destacar_regiao_doenca, compor_diagnostico_visual,
    aplicar_blur_em_roi,
)
from yolo_detection import (
    carregar_modelo_classificacao, classificar_folha,
    classificar_lote, comparar_metodos,
)
from utils import redimensionar

plt.rcParams.update({'figure.dpi': 100, 'figure.facecolor': 'white'})

TAMANHO = (256, 256)
RESULTS_DIR = os.path.abspath(os.path.join('..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

def grade(imagens, titulos, cols=3, figsize=(15, 5), nome=None, dpi=100):
    n = len(imagens)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            img = imagens[i]
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(titulos[i], fontsize=10, pad=4)
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    if nome:
        caminho = os.path.join(RESULTS_DIR, nome)
        fig.savefig(caminho, bbox_inches='tight', dpi=dpi)
        print(f'  Salvo -> results/{nome}')
    plt.show()
    plt.close(fig)

print('Setup completo.')
print(f'Resultados em: {RESULTS_DIR}')
"""),

# ── Dataset ───────────────────────────────────────────────────────────────────
M("---\n## 1 · Dataset PlantVillage"),

C("""
assert verificar_dataset(), (
    'Dataset nao encontrado.\\n'
    'Baixe em https://www.kaggle.com/datasets/mohitsingh1804/plantvillage\\n'
    'Extraia em data/PlantVillage/'
)

stats = estatisticas_dataset()
classes = listar_classes()   # usa train/ por padrao
saud, doentes = separar_saudaveis_doentes(classes)

print(f"Classes  : {stats['n_classes']}")
print(f"Train    : {stats['total_train']:,} imagens")
if stats['total_val']:
    print(f"Val      : {stats['total_val']:,} imagens")
print(f"Saudaveis: {len(saud)}")
print(f"Doencas  : {len(doentes)}")
print()
print('Distribuicao train (primeiras 12):')
for cls, n in list(stats['por_classe'].items())[:12]:
    barra = 'x' * (n // 100)
    print(f"  {nome_legivel(cls):<42} {n:>4}  {barra}")
"""),

C("""
# Seleciona par saudavel/doente (de train) que sera usado em todo o pipeline
for planta in ['Tomato', 'Apple', 'Potato', 'Corn', 'Grape']:
    pares = amostrar_pares(plantas=[planta], n=1, split='train', semente=7)
    if pares:
        break
if not pares:
    pares = amostrar_pares(n=1, split='train', semente=7)

par = pares[0]
IMG_SAUD  = cv2.resize(par['img_saudavel'], TAMANHO)
IMG_DOENT = cv2.resize(par['img_doente'],   TAMANHO)
NOME_SAUD  = nome_legivel(par['classe_saudavel'])
NOME_DOENT = nome_legivel(par['classe_doente'])

print(f"Planta  : {par['planta']}")
print(f"Saudavel: {NOME_SAUD}")
print(f"Doente  : {NOME_DOENT}")
"""),

C("""
# Amostras de 6 classes para visualizacao inicial (3 saudaveis + 3 doentes)
amostras = amostrar_por_classe(n=1, caminho=PASTA_DATASET, split='train', semente=0)
sel_s = [a for a in amostras if 'healthy'     in a['classe']][:3]
sel_d = [a for a in amostras if 'healthy' not in a['classe']][:3]
sel   = sel_s + sel_d

grade(
    [cv2.resize(a['img'], TAMANHO) for a in sel],
    [nome_legivel(a['classe']) for a in sel],
    cols=3, figsize=(14, 8),
    nome='00_amostras_dataset.png',
)
"""),

# ── Bloco 1: Blur + Threshold ─────────────────────────────────────────────────
M("""---
## 2 · Bloco 1 — Threshold & Blur

Aplicamos suavização para reduzir ruído e limiarização para segmentar regiões.
Comparamos os principais métodos de blur (Gaussian, Median, Bilateral) e limiarização (Simples, Otsu, Adaptativo)."""),

C("""
gray = converter_grayscale(IMG_DOENT)

b_gauss  = aplicar_blur_gaussiano(gray, kernel=(7, 7))
b_median = aplicar_blur_mediano(gray, ksize=7)
b_bilat  = aplicar_blur_bilateral(IMG_DOENT)  # RGB — preserva contornos coloridos

grade(
    [IMG_DOENT, gray, b_gauss, b_median, b_bilat],
    ['Original (RGB)', 'Grayscale',
     'Gaussian Blur (7x7)', 'Median Blur (k=7)', 'Bilateral (preserva bordas)'],
    cols=5, figsize=(20, 4),
    nome='01_blur_comparacao.png',
)
"""),

C("""
gray_blur = aplicar_blur_gaussiano(gray, kernel=(5, 5))
t_simples = aplicar_threshold_simples(gray_blur, valor=127)
t_otsu    = aplicar_threshold_otsu(gray_blur)
t_adapt   = aplicar_threshold_adaptativo_gaussiano(gray_blur, block_size=15, C=4)

grade(
    [gray_blur, t_simples, t_otsu, t_adapt],
    ['Grayscale + Blur', 'Simples (127)', 'Otsu (automatico)', 'Adaptativo Gaussiano'],
    cols=4, figsize=(16, 4),
    nome='02_threshold_comparacao.png',
)
"""),

C("""
gray_s = converter_grayscale(IMG_SAUD)
otsu_s = aplicar_threshold_otsu(aplicar_blur_gaussiano(gray_s))
otsu_d = aplicar_threshold_otsu(aplicar_blur_gaussiano(gray))

grade(
    [IMG_SAUD, otsu_s, IMG_DOENT, otsu_d],
    [f'Saudavel — {NOME_SAUD}', 'Otsu — Saudavel',
     f'Doente — {NOME_DOENT}',  'Otsu — Doente'],
    cols=4, figsize=(16, 4),
    nome='03_saudavel_vs_doente_threshold.png',
)
print(f"Pixels brancos (Otsu)  Saudavel: {otsu_s.sum() // 255:,}")
print(f"Pixels brancos (Otsu)  Doente  : {otsu_d.sum() // 255:,}")
"""),

# ── Bloco 2: Edge + Corner ────────────────────────────────────────────────────
M("""---
## 3 · Bloco 2 — Edge & Corner Detection

Detectamos bordas (Canny, Sobel) e cantos (Harris, Shi-Tomasi) para identificar
estruturas relevantes: bordas das lesões, nervuras e contornos da folha."""),

C("""
canny_s = detectar_bordas_canny(IMG_SAUD)
canny_d = detectar_bordas_canny(IMG_DOENT)

grade(
    [IMG_SAUD, canny_s, IMG_DOENT, canny_d],
    ['Saudavel', 'Canny — Saudavel', 'Doente', 'Canny — Doente'],
    cols=4, figsize=(16, 4),
    nome='04_canny.png',
)
print(f"Pixels de borda (Canny)  Saudavel: {(canny_s > 0).sum():,}  |  Doente: {(canny_d > 0).sum():,}")
"""),

C("""
mag, sx, sy = detectar_bordas_sobel(IMG_DOENT)

grade(
    [IMG_DOENT, sx, sy, mag],
    ['Original', 'Sobel X (bordas verticais)', 'Sobel Y (bordas horizontais)', 'Magnitude'],
    cols=4, figsize=(16, 4),
    nome='05_sobel.png',
)
"""),

C("""
img_harris,    n_h = detectar_cantos_harris(IMG_DOENT)
img_shitomasi, n_s = detectar_cantos_shi_tomasi(IMG_DOENT, max_cantos=80)

grade(
    [IMG_DOENT, img_harris, img_shitomasi],
    ['Original', f'Harris ({n_h} pontos)', f'Shi-Tomasi ({n_s} cantos)'],
    cols=3, figsize=(14, 5),
    nome='06_cantos.png',
)
"""),

C("""
cont_s, n_cs = detectar_contornos_externos(IMG_SAUD)
cont_d, n_cd = detectar_contornos_externos(IMG_DOENT)

grade(
    [cont_s, cont_d],
    [f'Contornos — Saudavel ({n_cs})', f'Contornos — Doente ({n_cd})'],
    cols=2, figsize=(12, 5),
    nome='07_contornos.png',
)
print(f"Contornos  Saudavel: {n_cs}  |  Doente: {n_cd}")
"""),

# ── Bloco 3: Detection + ROI + Blend/Paste ────────────────────────────────────
M("""---
## 4 · Bloco 3 — Detecção de Objeto + ROI + Blend/Paste

Segmentação por cor no espaço HSV para detectar a folha e as manchas de doença.
Composição visual com máscara binária, blend colorido e ROI com bounding boxes."""),

C("""
folha_seg, mascara_folha = segmentar_folha_verde(IMG_DOENT)

grade(
    [IMG_DOENT, mascara_folha, folha_seg],
    ['Original', 'Mascara binaria (folha verde)', 'Folha segmentada'],
    cols=3, figsize=(14, 5),
    nome='08_segmentacao_folha.png',
)
"""),

C("""
mascara_lesao, img_lesao, n_lesoes, bboxes = detectar_lesoes_hsv(IMG_DOENT, sensibilidade='media')

grade(
    [IMG_DOENT, mascara_lesao, img_lesao],
    ['Original', 'Mascara de lesoes (HSV)', 'Regioes doentes isoladas'],
    cols=3, figsize=(14, 5),
    nome='09_lesoes_hsv.png',
)
print(f"Lesoes detectadas: {n_lesoes}")
"""),

C("""
blend_r = destacar_regiao_doenca(IMG_DOENT, mascara_lesao, cor_destaque=(255,  60,  60))
blend_y = destacar_regiao_doenca(IMG_DOENT, mascara_lesao, cor_destaque=(255, 220,   0), alpha=0.5)

grade(
    [IMG_DOENT, blend_r, blend_y],
    ['Original', 'Blend vermelho (lesoes)', 'Blend amarelo (lesoes)'],
    cols=3, figsize=(14, 5),
    nome='10_blend_paste.png',
)
"""),

C("""
diagnostico = compor_diagnostico_visual(IMG_DOENT, mascara_lesao, bboxes, NOME_DOENT)

img_roi_blur = IMG_DOENT.copy()
for (x, y, w, h) in bboxes:
    img_roi_blur = aplicar_blur_em_roi(img_roi_blur, x, y, w, h, ksize=(21, 21))

grade(
    [IMG_SAUD, IMG_DOENT, diagnostico, img_roi_blur],
    ['Saudavel (referencia)', 'Doente (original)',
     'Diagnostico (blend + ROI)', 'ROI Blur nas lesoes'],
    cols=4, figsize=(18, 5),
    nome='11_diagnostico_roi.png',
)
"""),

C("""
# Analise em lote: 4 doencas diferentes
amostras_d = [a for a in amostrar_por_classe(n=1, caminho=PASTA_DATASET, split='train')
              if 'healthy' not in a['classe']][:4]

imgs_b, tits_b = [], []
for am in amostras_d:
    img  = cv2.resize(am['img'], TAMANHO)
    mask, _, n, bbox = detectar_lesoes_hsv(img)
    diag = compor_diagnostico_visual(img, mask, bbox, nome_legivel(am['classe']))
    imgs_b.append(diag)
    tits_b.append(f"{nome_legivel(am['classe'])} ({n} lesoes)")

if imgs_b:
    grade(imgs_b, tits_b, cols=2, figsize=(14, 10), nome='12_batch_diagnostico.png')
"""),

# ── Bloco 4: YOLO ─────────────────────────────────────────────────────────────
M("""---
## 5 · Bloco 4 — YOLO

Usamos **YOLOv8n-cls** pré-treinado no ImageNet para classificação das folhas.
As predições mostram as classes mais prováveis segundo o modelo.
Para obter os nomes das doenças do PlantVillage, seria necessário fine-tuning
(disponível em `yolo_detection.treinar_classificador()`)."""),

C("""
modelo_cls = carregar_modelo_classificacao('yolov8n-cls.pt')
print('YOLOv8n-cls carregado.')
"""),

C("""
img_yolo_s, preds_s = classificar_folha(IMG_SAUD,  modelo_cls, top_k=5)
img_yolo_d, preds_d = classificar_folha(IMG_DOENT, modelo_cls, top_k=5)

grade(
    [IMG_SAUD, img_yolo_s, IMG_DOENT, img_yolo_d],
    ['Saudavel (original)', 'YOLO — Saudavel', 'Doente (original)', 'YOLO — Doente'],
    cols=4, figsize=(16, 4),
    nome='13_yolo_classificacao.png',
)

print('Top-3 predicoes — Saudavel:')
for p in preds_s[:3]:
    print(f"  {p['classe']:<30} {p['confianca']:.1%}")
print()
print('Top-3 predicoes — Doente:')
for p in preds_d[:3]:
    print(f"  {p['classe']:<30} {p['confianca']:.1%}")
"""),

C("""
batch = [
    dict(a, img=cv2.resize(a['img'], TAMANHO))
    for a in amostrar_por_classe(n=1, caminho=PASTA_DATASET, split='train')[:6]
]
resultados = classificar_lote(batch, modelo_cls)

imgs_yl = [r['img_yolo'] for r in resultados]
tits_yl = [
    f"{nome_legivel(r['classe'])[:22]} | YOLO: {r['predicao_yolo'][:18]} ({r['confianca_yolo']:.0%})"
    for r in resultados
]

grade(imgs_yl, tits_yl, cols=3, figsize=(18, 10), nome='14_yolo_lote.png')
"""),

# ── Comparação ────────────────────────────────────────────────────────────────
M("---\n## 6 · Comparação: Métodos Clássicos vs YOLO"),

C("""
comparacao = comparar_metodos(
    IMG_DOENT,
    diagnostico,
    img_yolo_d,
    label_classico='Classico (HSV + Blend + ROI)',
    label_yolo='YOLO (YOLOv8-cls)',
)

fig, ax = plt.subplots(figsize=(16, 5), dpi=100)
ax.imshow(comparacao)
ax.axis('off')
ax.set_title('Comparacao: Original  |  Classico  |  YOLO', fontsize=13, pad=8)
plt.tight_layout()
caminho = os.path.join(RESULTS_DIR, '15_comparacao_final.png')
fig.savefig(caminho, bbox_inches='tight', dpi=100)
print('  Salvo -> results/15_comparacao_final.png')
plt.show()
plt.close(fig)
"""),

C("""
print('''
Criterio                      | Classico (HSV + PDI)   | YOLO (YOLOv8-cls)
------------------------------|------------------------|---------------------
Precisa de treinamento?       | Nao                    | Sim (fine-tuning)
Localiza regioes de lesao?    | Sim (mascara + ROI)    | Nao (cls puro)
Velocidade de inferencia      | Muito rapida           | Rapida
Sensivel a iluminacao         | Sim (HSV fixo)         | Menos sensivel
Identifica nome da doenca     | Nao                    | Sim (com fine-tuning)
Interpretabilidade            | Alta (visual)          | Caixa-preta
Funciona sem GPU              | Sim                    | Sim (modo CPU)
''')
"""),

# ── Resumo dos arquivos ───────────────────────────────────────────────────────
M("---\n## 7 · Arquivos Gerados em `results/`"),

C("""
arquivos = sorted(
    f for f in os.listdir(RESULTS_DIR) if f.endswith(('.png', '.jpg'))
)

print(f"{'Arquivo':<48} {'Tamanho':>10}")
print('-' * 60)
for f in arquivos:
    tam = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1024
    print(f"  {f:<46} {tam:>6.1f} KB")
print()
print(f"Total: {len(arquivos)} arquivos em {RESULTS_DIR}")
"""),

]  # end cells

nb.cells = cells

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook gerado: {OUTPUT}")

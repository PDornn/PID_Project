"""
Geração de gráficos de avaliação do treinamento YOLOv8.
Lê results.csv e copia os artefatos do Ultralytics para output_dir.
"""
import csv
import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─── Leitura do CSV ───────────────────────────────────────────────────────────

def _ler_csv(results_csv):
    with open(results_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    dados = {}
    for key in rows[0]:
        dados[key.strip()] = [float(row[key]) for row in rows]
    return dados


# ─── Helpers de estilo ────────────────────────────────────────────────────────

_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.color':       '#dee2e6',
    'grid.linestyle':   '--',
    'grid.linewidth':   0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'font.size':          11,
}

def _salvar(fig, output_dir, nome, dpi=120, mostrar=False):
    os.makedirs(output_dir, exist_ok=True)
    caminho = os.path.join(output_dir, nome)
    fig.savefig(caminho, bbox_inches='tight', dpi=dpi)
    if mostrar:
        plt.show()
    plt.close(fig)
    print(f'  Salvo -> {caminho}')
    return caminho


# ─── 01 · Curvas de perda (train vs val) ─────────────────────────────────────

def plot_curvas_perda(dados, output_dir, dpi=120, mostrar=False):
    epocas = dados['epoch']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Curvas de Perda — YOLOv8n (30 épocas)', fontsize=13, y=1.02)

    pares = [
        ('train/box_loss', 'val/box_loss', 'Box Loss (regressão CIoU + DFL)'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss (BCE)'),
        ('train/dfl_loss', 'val/dfl_loss', 'Distribution Focal Loss'),
    ]
    with plt.rc_context(_STYLE):
        for ax, (t_col, v_col, titulo) in zip(axes, pares):
            ax.plot(epocas, dados[t_col], color='#2196F3', linewidth=2,
                    label='Treino', marker='o', markersize=3)
            ax.plot(epocas, dados[v_col], color='#FF5722', linewidth=2,
                    label='Validação', marker='s', markersize=3)
            ax.set_title(titulo, fontsize=11, pad=8)
            ax.set_xlabel('Época')
            ax.set_ylabel('Perda')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(framealpha=0.8)

    plt.tight_layout()
    return _salvar(fig, output_dir, '01_curvas_perda.png', dpi, mostrar)


# ─── 02 · Métricas de validação ──────────────────────────────────────────────

def plot_metricas_validacao(dados, output_dir, dpi=120, mostrar=False):
    epocas = dados['epoch']
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Métricas de Validação — YOLOv8n (30 épocas)', fontsize=13, y=1.01)

        metricas = [
            ('metrics/precision(B)', 'Precision', '#4CAF50'),
            ('metrics/recall(B)',    'Recall',    '#FF9800'),
            ('metrics/mAP50(B)',     'mAP@50',    '#9C27B0'),
            ('metrics/mAP50-95(B)', 'mAP@50-95', '#F44336'),
        ]
        for ax, (col, label, cor) in zip(axes.flatten(), metricas):
            vals = dados[col]
            idx_max = vals.index(max(vals))
            ax.plot(epocas, vals, color=cor, linewidth=2.2,
                    marker='o', markersize=3.5, label=label)
            ax.axhline(max(vals), color=cor, linewidth=0.8, linestyle='--', alpha=0.6)
            ax.annotate(
                f'máx: {max(vals):.4f}  (época {int(epocas[idx_max])})',
                xy=(epocas[idx_max], max(vals)),
                xytext=(6, -14), textcoords='offset points',
                fontsize=9, color=cor,
            )
            ax.set_title(label, fontsize=11, pad=8)
            ax.set_xlabel('Época')
            ax.set_ylabel(label)
            ax.set_ylim(max(0, min(vals) - 0.05), min(1.05, max(vals) + 0.06))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    return _salvar(fig, output_dir, '02_metricas_validacao.png', dpi, mostrar)


# ─── 03 · Taxa de aprendizado ────────────────────────────────────────────────

def plot_learning_rate(dados, output_dir, dpi=120, mostrar=False):
    epocas = dados['epoch']
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epocas, dados['lr/pg0'], color='#607D8B', linewidth=2,
                label='lr (todos os grupos)')
        ax.fill_between(epocas, dados['lr/pg0'], alpha=0.15, color='#607D8B')
        ax.set_title('Schedule de Taxa de Aprendizado (Cosine Annealing)', fontsize=12)
        ax.set_xlabel('Época')
        ax.set_ylabel('Learning Rate')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
        ax.legend()
        plt.tight_layout()
    return _salvar(fig, output_dir, '03_learning_rate.png', dpi, mostrar)


# ─── 04 · Resumo final (barras) ──────────────────────────────────────────────

def plot_resumo_final(dados, output_dir, dpi=120, mostrar=False):
    ultima = {k: v[-1] for k, v in dados.items()}
    melhor_ep = int(dados['epoch'][
        dados['metrics/mAP50(B)'].index(max(dados['metrics/mAP50(B)']))])

    labels  = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95']
    cols    = ['metrics/precision(B)', 'metrics/recall(B)',
               'metrics/mAP50(B)',     'metrics/mAP50-95(B)']
    cores   = ['#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    valores = [ultima[c] for c in cols]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, valores, color=cores, width=0.5,
                      edgecolor='white', linewidth=1.5, zorder=3)
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Valor (escala 0–1)')
        ax.set_title(
            f'Métricas finais — época 30  |  Melhor mAP50 na época {melhor_ep}',
            fontsize=12, pad=10)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        plt.tight_layout()
    return _salvar(fig, output_dir, '04_resumo_final.png', dpi, mostrar)


# ─── 05-10 · Copiar + exibir artefatos do Ultralytics ────────────────────────

def copiar_artefatos(run_dir, output_dir, mostrar=False):
    artefatos = {
        'confusion_matrix_normalized.png': '05_confusion_matrix_norm.png',
        'confusion_matrix.png':            '06_confusion_matrix.png',
        'BoxPR_curve.png':                 '07_pr_curve.png',
        'BoxF1_curve.png':                 '08_f1_curve.png',
        'BoxP_curve.png':                  '09_precision_curve.png',
        'BoxR_curve.png':                  '10_recall_curve.png',
    }
    os.makedirs(output_dir, exist_ok=True)
    copiados = []
    for origem_nome, destino_nome in artefatos.items():
        origem  = os.path.join(run_dir, origem_nome)
        destino = os.path.join(output_dir, destino_nome)
        if not os.path.exists(origem):
            continue
        shutil.copy2(origem, destino)
        print(f'  Copiado -> {destino}')
        copiados.append(destino)
        if mostrar:
            img = cv2.cvtColor(cv2.imread(destino), cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img); ax.axis('off')
            ax.set_title(destino_nome.replace('_', ' ').replace('.png', ''), fontsize=12)
            plt.tight_layout(); plt.show(); plt.close(fig)
    return copiados


# ─── 11 · Batch de validação lado a lado ─────────────────────────────────────

def plot_val_batch(run_dir, output_dir, dpi=120, mostrar=False):
    labels_path = os.path.join(run_dir, 'val_batch0_labels.jpg')
    pred_path   = os.path.join(run_dir, 'val_batch0_pred.jpg')
    if not (os.path.exists(labels_path) and os.path.exists(pred_path)):
        return None

    labels_img = cv2.cvtColor(cv2.imread(labels_path), cv2.COLOR_BGR2RGB)
    pred_img   = cv2.cvtColor(cv2.imread(pred_path),   cv2.COLOR_BGR2RGB)
    h = max(labels_img.shape[0], pred_img.shape[0])
    w = max(labels_img.shape[1], pred_img.shape[1])
    labels_img = cv2.resize(labels_img, (w, h))
    pred_img   = cv2.resize(pred_img,   (w, h))

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Batch de Validação — Ground-Truth vs Predições YOLO', fontsize=13)
        axes[0].imshow(labels_img); axes[0].set_title('Ground-Truth',   fontsize=11); axes[0].axis('off')
        axes[1].imshow(pred_img);   axes[1].set_title('Predições YOLOv8n', fontsize=11); axes[1].axis('off')
        plt.tight_layout()
    return _salvar(fig, output_dir, '11_val_batch_comparacao.png', dpi, mostrar)


# ─── Pipeline completo ────────────────────────────────────────────────────────

def gerar_graficos(results_csv, run_dir, output_dir, dpi=120, mostrar=False):
    """
    Gera todos os gráficos de avaliação, salva em output_dir e,
    se mostrar=True, exibe cada figura inline (uso no notebook).

    Parâmetros:
        results_csv — caminho do results.csv do Ultralytics
        run_dir     — pasta da run (confusion_matrix.png etc.)
        output_dir  — pasta de destino (ex: results/treinamento)
        dpi         — resolução dos arquivos de saída
        mostrar     — True para exibir inline no notebook
    """
    print(f'\nGerando gráficos em: {output_dir}')
    dados = _ler_csv(results_csv)

    plot_curvas_perda(dados, output_dir, dpi, mostrar)
    plot_metricas_validacao(dados, output_dir, dpi, mostrar)
    plot_learning_rate(dados, output_dir, dpi, mostrar)
    plot_resumo_final(dados, output_dir, dpi, mostrar)
    copiar_artefatos(run_dir, output_dir, mostrar)
    plot_val_batch(run_dir, output_dir, dpi, mostrar)

    arquivos = sorted(f for f in os.listdir(output_dir)
                      if f.endswith(('.png', '.jpg')))
    print(f'\n{len(arquivos)} arquivo(s) em {output_dir}:')
    for f in arquivos:
        tam = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f'  {f:<46} {tam:>6.1f} KB')

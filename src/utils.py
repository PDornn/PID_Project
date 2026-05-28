"""
Utilitários de exibição e salvamento - Capstone de Processamento Digital de Imagens
"""
import cv2
import matplotlib.pyplot as plt
import os

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RESULTS = os.path.join(PROJETO_RAIZ, 'results')


def mostrar_imagem(img, titulo='Imagem', figsize=(8, 6)):
    """Exibe uma única imagem com matplotlib."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(titulo, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def mostrar_comparacao(imagens, titulos, cols=3, figsize=(15, 5)):
    """
    Exibe múltiplas imagens em grade para comparação visual.

    Parâmetros:
        imagens - lista de imagens numpy
        titulos - lista de títulos correspondentes
        cols    - número de colunas na grade
        figsize - tamanho total da figura
    """
    n = len(imagens)
    rows = (n + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=figsize)

    # Normaliza axes para sempre ser array 1D
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    elif cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = list(axes.flatten())

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n:
            img = imagens[i]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(titulos[i], fontsize=11, pad=5)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def salvar_resultado(img, nome_arquivo, pasta=None):
    """
    Salva uma imagem na pasta /results do projeto.

    Converte de RGB para BGR antes de salvar (padrão OpenCV).
    """
    if pasta is None:
        pasta = PASTA_RESULTS

    os.makedirs(pasta, exist_ok=True)

    if len(img.shape) == 3:
        img_salvar = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_salvar = img.copy()

    caminho = os.path.join(pasta, nome_arquivo)
    cv2.imwrite(caminho, img_salvar)
    print(f'[Utils] Salvo: {caminho}')


def redimensionar(img, largura=None, altura=None):
    """
    Redimensiona a imagem mantendo a proporção original.
    Informe apenas largura OU altura para manter o aspecto.
    """
    h, w = img.shape[:2]

    if largura is None and altura is None:
        return img

    if largura is None:
        escala = altura / h
        largura = int(w * escala)
    elif altura is None:
        escala = largura / w
        altura = int(h * escala)

    return cv2.resize(img, (largura, altura))


def desenhar_texto(img, texto, posicao=(10, 30), escala=1.0, cor=(0, 255, 0), espessura=2):
    """Escreve texto sobre a imagem (copia)."""
    img_copia = img.copy()
    cv2.putText(img_copia, texto, posicao,
                cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura)
    return img_copia


def mostrar_histograma(img, titulo='Histograma'):
    """Exibe o histograma de uma imagem em escala de cinza."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    plt.figure(figsize=(8, 4))
    plt.hist(gray.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.8)
    plt.title(titulo, fontsize=14)
    plt.xlabel('Intensidade do Pixel')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()

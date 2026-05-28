"""
Pré-processamento de Imagens - Capstone de Processamento Digital de Imagens

Técnicas vistas em aula:
  - processamento_de_imagens/00-Color-Mappings.ipynb
  - processamento_de_imagens/02-Image-Thresholding.ipynb
  - processamento_de_imagens/03-Blurring-and-Smoothing.ipynb
"""
import cv2
import numpy as np


# ─── Conversão de cor ────────────────────────────────────────────────────────

def converter_grayscale(img):
    """Converte imagem RGB para escala de cinza."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def converter_hsv(img):
    """Converte imagem RGB para espaço de cor HSV."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


# ─── Suavização / Blur ───────────────────────────────────────────────────────

def aplicar_blur_gaussiano(img, kernel=(5, 5), sigma=0):
    """
    Gaussian Blur: suaviza a imagem com uma função gaussiana.
    Reduz ruído e detalhes finos antes de aplicar outras técnicas.
    """
    return cv2.GaussianBlur(img, kernel, sigma)


def aplicar_blur_mediano(img, ksize=5):
    """
    Median Blur: substitui cada pixel pela mediana da vizinhança.
    Excelente para remover ruído do tipo 'sal e pimenta'.
    """
    return cv2.medianBlur(img, ksize)


def aplicar_blur_bilateral(img, d=9, sigma_cor=75, sigma_espaco=75):
    """
    Bilateral Filter: suaviza preservando as bordas da imagem.
    Mais lento, mas mantém contornos nítidos.
    """
    return cv2.bilateralFilter(img, d, sigma_cor, sigma_espaco)


def aplicar_filtro_2d(img, tamanho_kernel=5):
    """
    Filtro 2D com kernel de média (passa-baixa via convolução).
    Técnica vista em aula de Blurring and Smoothing.
    """
    kernel = np.ones((tamanho_kernel, tamanho_kernel), dtype=np.float32)
    kernel = kernel / (tamanho_kernel ** 2)
    return cv2.filter2D(img, -1, kernel)


# ─── Threshold (Limiarização) ────────────────────────────────────────────────

def aplicar_threshold_simples(img_gray, valor=127):
    """
    Threshold binário simples.
    Pixels acima de 'valor' -> branco (255); abaixo -> preto (0).
    """
    _, thresh = cv2.threshold(img_gray, valor, 255, cv2.THRESH_BINARY)
    return thresh


def aplicar_threshold_otsu(img_gray):
    """
    Threshold de Otsu: calcula automaticamente o limiar ideal.
    Ótimo quando não sabemos qual valor usar — a imagem decide.
    """
    _, thresh = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def aplicar_threshold_adaptativo_media(img_gray, block_size=11, C=8):
    """
    Threshold adaptativo com média local.
    Calcula o limiar para cada região da imagem separadamente.
    Útil para imagens com iluminação não uniforme.
    """
    return cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )


def aplicar_threshold_adaptativo_gaussiano(img_gray, block_size=11, C=2):
    """
    Threshold adaptativo com pesos gaussianos.
    Versão mais suave do adaptativo — geralmente produz melhores resultados.
    """
    return cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )


# ─── Ajustes de brilho ───────────────────────────────────────────────────────

def aplicar_gamma(img, gamma=1.0):
    """
    Correção gamma para ajustar o brilho da imagem.
    gamma < 1.0 -> clareia; gamma > 1.0 -> escurece.
    Técnica vista em aula de Blurring and Smoothing.
    """
    img_float = img.astype(np.float32) / 255.0
    corrigida = np.power(img_float, gamma)
    return np.clip(corrigida * 255, 0, 255).astype(np.uint8)

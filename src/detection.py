"""
Detecção Clássica de Imagens - Capstone de Processamento Digital de Imagens

Técnicas vistas em aula:
  - deteccao_de_imagens/02-Edge-Detection.ipynb      (Canny, Sobel)
  - deteccao_de_imagens/01-Corner-Detection.ipynb    (Harris, Shi-Tomasi)
  - deteccao_de_imagens/04-Contour-Detection.ipynb   (Contornos)
  - deteccao_de_imagens/08-Face-Detection.ipynb      (Haar Cascade)
  - processamento_de_imagens/01-Blending-and-Pasting-Images.ipynb (ROI/Máscara)

Funções adicionais para detecção de doenças em plantas (PlantVillage):
  - detectar_bordas_sobel         : gradiente de intensidade via Sobel
  - segmentar_folha_verde         : isola a folha do fundo via HSV
  - detectar_lesoes_hsv           : segmenta manchas de doença (marrom/amarelo)
  - destacar_regiao_doenca        : aplica máscara colorida sobre lesões
  - compor_diagnostico_visual     : composição final blend/paste com ROI
"""
import cv2
import numpy as np
import os

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURSO_RAIZ = os.path.dirname(os.path.dirname(PROJETO_RAIZ))
HAARCASCADES = os.path.join(CURSO_RAIZ, 'deteccao_de_imagens', 'haarcascades')


# ─── Detecção de Bordas ──────────────────────────────────────────────────────

def detectar_bordas_canny(img, threshold1=None, threshold2=None):
    """
    Detecção de bordas com o algoritmo Canny.

    Se os thresholds não forem informados, são calculados automaticamente
    com base na mediana dos pixels — técnica ensinada em aula.

    Etapas:
      1. Converte para grayscale
      2. Aplica blur para reduzir bordas falsas
      3. Aplica Canny com thresholds baseados na mediana
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Blur antes do Canny para suavizar ruído (ensinado em aula)
    blurred = cv2.blur(gray, ksize=(5, 5))

    if threshold1 is None or threshold2 is None:
        # Thresholds automáticos baseados na mediana dos pixels
        med = np.median(blurred)
        threshold1 = int(max(0, 0.7 * med))
        threshold2 = int(min(255, 1.3 * med))

    edges = cv2.Canny(image=blurred, threshold1=threshold1, threshold2=threshold2)
    return edges


# ─── Detecção de Cantos ──────────────────────────────────────────────────────

def detectar_cantos_harris(img):
    """
    Detecção de cantos com o algoritmo Harris.
    Os cantos detectados são marcados em vermelho na imagem.

    Parâmetros do cornerHarris (vistos em aula):
      - blockSize=2 : tamanho da vizinhança para o cálculo
      - ksize=3     : abertura do operador Sobel
      - k=0.04      : parâmetro livre do detector Harris
    """
    img_copia = img.copy()
    if len(img_copia.shape) == 2:
        img_copia = cv2.cvtColor(img_copia, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(img_copia, cv2.COLOR_RGB2GRAY)

    # Harris exige float32
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(src=gray_float, blockSize=2, ksize=3, k=0.04)

    # Dilata para facilitar a visualização dos pontos
    dst = cv2.dilate(dst, None)

    # Marca cantos em vermelho onde a resposta é forte
    img_copia[dst > 0.01 * dst.max()] = [255, 0, 0]

    n_cantos = int(np.sum(dst > 0.01 * dst.max()))
    return img_copia, n_cantos


def detectar_cantos_shi_tomasi(img, max_cantos=50, qualidade=0.01, distancia=10):
    """
    Detecção de cantos com Shi-Tomasi (goodFeaturesToTrack).
    Cantos marcados como círculos verdes na imagem.

    Parâmetros (vistos em aula):
      - max_cantos : número máximo de cantos a retornar
      - qualidade  : qualidade mínima dos cantos (0–1)
      - distancia  : distância mínima entre cantos em pixels
    """
    img_copia = img.copy()
    if len(img_copia.shape) == 2:
        img_copia = cv2.cvtColor(img_copia, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(img_copia, cv2.COLOR_RGB2GRAY)

    cantos = cv2.goodFeaturesToTrack(gray, max_cantos, qualidade, distancia)

    if cantos is not None:
        cantos = cantos.astype(int)
        for c in cantos:
            x, y = c.ravel()
            cv2.circle(img_copia, (x, y), 5, (0, 255, 0), -1)
        return img_copia, len(cantos)

    return img_copia, 0


# ─── Detecção de Contornos ───────────────────────────────────────────────────

def detectar_contornos_externos(img):
    """
    Detecta e desenha contornos externos na imagem.

    Etapas (vistas em aula — Contour Detection):
      1. Converte para grayscale
      2. Aplica threshold para binarizar
      3. Encontra contornos com RETR_EXTERNAL
      4. Desenha apenas os contornos externos (em verde)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # findContours retorna (contornos, hierarquia) no OpenCV 4+
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_resultado = img.copy()
    if len(img_resultado.shape) == 2:
        img_resultado = cv2.cvtColor(img_resultado, cv2.COLOR_GRAY2RGB)

    # -1 para desenhar todos; 2 é a espessura
    cv2.drawContours(img_resultado, contours, -1, (0, 255, 0), 2)

    return img_resultado, len(contours)


def detectar_contornos_internos_externos(img):
    """
    Detecta contornos internos e externos separadamente.
    Técnica diretamente do notebook 04-Contour-Detection das aulas.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Imagens vazias para desenhar separadamente
    externos = np.zeros(gray.shape, dtype=np.uint8)
    internos = np.zeros(gray.shape, dtype=np.uint8)

    for i in range(len(contours)):
        # hierarchy[0][i][3] == -1 significa contorno externo
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(externos, contours, i, 255, -1)
        else:
            cv2.drawContours(internos, contours, i, 255, -1)

    return externos, internos


# ─── Detecção de Faces com Haar Cascade ─────────────────────────────────────

def _carregar_cascade(nome_arquivo):
    """Tenta carregar o cascade das aulas; usa o do OpenCV como fallback."""
    caminho_aulas = os.path.join(HAARCASCADES, nome_arquivo)
    if os.path.exists(caminho_aulas):
        return cv2.CascadeClassifier(caminho_aulas)

    # Fallback: cascade embutido no OpenCV
    caminho_opencv = cv2.data.haarcascades + nome_arquivo
    return cv2.CascadeClassifier(caminho_opencv)


def detectar_faces_haar(img, scale_factor=1.1, min_neighbors=5):
    """
    Detecta rostos humanos usando Haar Cascade.
    Técnica vista em aula: deteccao_de_imagens/08-Face-Detection.

    Parâmetros:
      - scale_factor  : fator de escala entre níveis da pirâmide
      - min_neighbors : número mínimo de vizinhos para confirmar detecção
    """
    face_cascade = _carregar_cascade('haarcascade_frontalface_default.xml')

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    img_resultado = img.copy()
    if len(img_resultado.shape) == 2:
        img_resultado = cv2.cvtColor(img_resultado, cv2.COLOR_GRAY2RGB)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return img_resultado, len(faces)


def detectar_olhos_haar(img):
    """Detecta olhos usando Haar Cascade — técnica vista em aula."""
    eye_cascade = _carregar_cascade('haarcascade_eye.xml')

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    olhos = eye_cascade.detectMultiScale(gray)

    img_resultado = img.copy()
    if len(img_resultado.shape) == 2:
        img_resultado = cv2.cvtColor(img_resultado, cv2.COLOR_GRAY2RGB)

    for (x, y, w, h) in olhos:
        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return img_resultado, len(olhos)


# ─── ROI e Máscara ───────────────────────────────────────────────────────────

def aplicar_blur_em_roi(img, x, y, w, h, ksize=(31, 31)):
    """
    Aplica Gaussian Blur somente dentro de uma ROI (Região de Interesse).
    Útil para anonimizar faces detectadas automaticamente.
    Conceito de ROI visto em aula: 01-Blending-and-Pasting-Images.
    """
    img_resultado = img.copy()
    roi = img_resultado[y:y + h, x:x + w]
    roi_blur = cv2.GaussianBlur(roi, ksize, 0)
    img_resultado[y:y + h, x:x + w] = roi_blur
    return img_resultado


def anonimizar_faces(img):
    """Detecta faces e aplica blur em cada uma (pipeline combinado)."""
    face_cascade = _carregar_cascade('haarcascade_frontalface_default.xml')

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    img_resultado = img.copy()
    for (x, y, w, h) in faces:
        img_resultado = aplicar_blur_em_roi(img_resultado, x, y, w, h)

    return img_resultado, len(faces)


# ─── Detecção de Bordas com Sobel ────────────────────────────────────────────

def detectar_bordas_sobel(img, ksize=3):
    """
    Calcula o gradiente de intensidade via operador Sobel.

    Aplica Sobel em X e Y separadamente e combina com a magnitude euclidiana.
    Útil para visualizar a estrutura das nervuras e bordas das lesões nas folhas.

    Parâmetros:
        img   - imagem RGB ou grayscale
        ksize - tamanho do kernel Sobel (1, 3, 5 ou 7)

    Retorna:
        magnitude - imagem uint8 com magnitude do gradiente
        sobel_x   - gradiente horizontal (uint8)
        sobel_y   - gradiente vertical (uint8)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)
    sx = cv2.convertScaleAbs(sobelx)
    sy = cv2.convertScaleAbs(sobely)

    return magnitude, sx, sy


# ─── Segmentação de Planta / Doença por Cor (HSV) ────────────────────────────

def segmentar_folha_verde(img_rgb):
    """
    Isola a folha do fundo usando segmentação por cor no espaço HSV.

    Aplica uma máscara para manter apenas os tons de verde típicos de folhas.
    Útil como pré-etapa antes de detectar as lesões.

    Retorna:
        folha_segmentada - imagem RGB com o fundo removido (preto)
        mascara          - máscara binária da folha (uint8)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Verde claro e verde escuro cobrem a maioria das folhas saudáveis
    lower_verde = np.array([25, 40, 40])
    upper_verde = np.array([90, 255, 255])
    mascara = cv2.inRange(hsv, lower_verde, upper_verde)

    # Morfologia para fechar buracos e remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    folha_segmentada = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara)
    return folha_segmentada, mascara


def detectar_lesoes_hsv(img_rgb, sensibilidade='media'):
    """
    Segmenta manchas de doença (tons marrom, amarelo, bege) usando HSV.

    Doenças como Early Blight, Late Blight e Bacterial Spot produzem lesões
    com cores que diferem do verde saudável. Este método as isola via HSV.

    Parâmetros:
        img_rgb      - imagem RGB da folha
        sensibilidade - 'baixa', 'media' ou 'alta' (controla abertura da faixa HSV)

    Retorna:
        mascara_lesao  - máscara binária com as regiões de lesão (uint8)
        img_lesao      - imagem RGB com apenas as regiões de lesão visíveis
        n_lesoes       - número de componentes conectados de lesão detectados
        bboxes         - lista de (x, y, w, h) de cada lesão
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Faixas HSV para lesões: marrom, amarelo-torrado, bege
    faixas = {
        'baixa': [
            (np.array([10, 60, 40]),  np.array([25, 255, 200])),   # marrom escuro
            (np.array([20, 50, 150]), np.array([35, 255, 255])),   # amarelo-pálido
        ],
        'media': [
            (np.array([5,  40,  30]),  np.array([30, 255, 220])),  # marrom
            (np.array([15, 30, 100]), np.array([40, 255, 255])),   # amarelo/bege
            (np.array([0,  20,  60]),  np.array([12, 200, 200])),  # tons alaranjados
        ],
        'alta': [
            (np.array([0,  20,  20]),  np.array([40, 255, 255])),  # marrom a amarelo amplo
            (np.array([0,  10,  30]),  np.array([15, 180, 180])),  # tons escuros
        ],
    }

    combinada = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in faixas.get(sensibilidade, faixas['media']):
        combinada = cv2.bitwise_or(combinada, cv2.inRange(hsv, lower, upper))

    # Morfologia para remover ruído e conectar manchas próximas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combinada = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel, iterations=1)
    combinada = cv2.morphologyEx(combinada, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Componentes conectados para contar e localizar lesões
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combinada)
    area_minima = 50
    bboxes = []
    mascara_final = np.zeros_like(combinada)

    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_minima:
            mascara_final[labels == i] = 255
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bboxes.append((x, y, w, h))

    img_lesao = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara_final)
    return mascara_final, img_lesao, len(bboxes), bboxes


def destacar_regiao_doenca(img_rgb, mascara_lesao, cor_destaque=(255, 80, 80), alpha=0.55):
    """
    Aplica uma sobreposição colorida (blend) sobre as regiões de lesão detectadas.

    Técnica de composição visual: blend entre a imagem original e uma camada de cor
    usando a máscara binária — equivalente ao blend/paste ensinado em aula.

    Parâmetros:
        img_rgb       - imagem original RGB
        mascara_lesao - máscara binária das lesões (uint8)
        cor_destaque  - cor RGB do destaque (padrão: vermelho)
        alpha         - opacidade do destaque (0=transparente, 1=sólido)

    Retorna imagem RGB com lesões destacadas.
    """
    destaque = np.zeros_like(img_rgb)
    destaque[:] = cor_destaque

    # Blend apenas nas regiões da máscara
    mascara_3c = cv2.merge([mascara_lesao, mascara_lesao, mascara_lesao]).astype(np.float32) / 255.0
    img_float = img_rgb.astype(np.float32)
    dest_float = destaque.astype(np.float32)

    blendado = img_float * (1 - alpha * mascara_3c) + dest_float * (alpha * mascara_3c)
    return np.clip(blendado, 0, 255).astype(np.uint8)


def compor_diagnostico_visual(img_rgb, mascara_lesao, bboxes, label_classe=''):
    """
    Cria uma composição de diagnóstico visual completa.

    Combina:
      - Destaque colorido (blend) nas lesões
      - Bounding boxes ao redor de cada lesão (ROI)
      - Texto informativo com a classe e contagem de lesões

    Parâmetros:
        img_rgb       - imagem original RGB
        mascara_lesao - máscara binária das lesões
        bboxes        - lista de (x, y, w, h) de cada lesão
        label_classe  - nome da doença para exibir na imagem

    Retorna imagem RGB final de diagnóstico.
    """
    resultado = destacar_regiao_doenca(img_rgb, mascara_lesao)

    for (x, y, w, h) in bboxes:
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (255, 220, 0), 2)

    n = len(bboxes)
    info = f'{label_classe}  |  {n} lesao{"es" if n != 1 else ""} detectada{"s" if n != 1 else ""}'
    h_img = resultado.shape[0]
    cv2.rectangle(resultado, (0, h_img - 32), (resultado.shape[1], h_img), (20, 20, 20), -1)
    cv2.putText(resultado, info, (8, h_img - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return resultado

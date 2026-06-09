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
  - segmentar_folha_verde         : isola a folha do fundo (BORDA via CANNY)
  - detectar_lesoes_hsv           : segmenta manchas de doença (marrom/amarelo)
  - destacar_regiao_doenca        : aplica máscara colorida sobre lesões
  - compor_diagnostico_visual     : composição final blend/paste com ROI

NOTA SOBRE O CANNY (correção principal):
  O Canny agora é o MOTOR da segmentação da folha. O fundo texturizado é
  suprimido por uma pré-máscara de cor (verde via canal 'a' do LAB ⋃ marrom
  via HSV); o Canny detecta a borda real da folha dentro dessa região; essa
  borda é fechada e preenchida para virar a máscara. A pré-máscara também
  "grampeia" o resultado para o contorno colar nas serrilhas e não estourar
  para o fundo. As lesões de margem entram na folha porque o marrom participa
  da pré-máscara.
"""
import cv2
import numpy as np
import os

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURSO_RAIZ = os.path.dirname(os.path.dirname(PROJETO_RAIZ))
HAARCASCADES = os.path.join(CURSO_RAIZ, 'deteccao_de_imagens', 'haarcascades')


# ─── Helpers compartilhados ──────────────────────────────────────────────────

# Faixas HSV de marrom/necrótico calibradas nos pixels reais da amostra de uva
# (OpenCV: H 0-179). Lesão típica medida: H≈13, S≈150, V≈52 (marrom escuro saturado).
_FAIXAS_MARROM = [
    (np.array([0, 40, 20]), np.array([25, 255, 190])),   # corpo marrom
    (np.array([0, 25, 20]), np.array([20, 255, 120])),   # marrom escuro de margem
]


def _auto_canny(gray, sigma=0.33):
    """Canny com limiares automáticos pela mediana (ignora pixels zerados/fundo)."""
    amostra = gray[gray > 0]
    med = np.median(amostra) if amostra.size else 0
    t1 = int(max(0, (1.0 - sigma) * med))
    t2 = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(gray, t1, t2)


def _mascara_marrom(hsv):
    """Une as faixas HSV de marrom em uma máscara binária (verde NÃO entra)."""
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _FAIXAS_MARROM:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
    return m


def _premask_cor(img_rgb):
    """
    Pré-máscara de COR = folha verde (canal 'a' do LAB, Otsu) ⋃ manchas marrons (HSV).

    Serve para (a) suprimir o fundo texturizado antes do Canny e (b) garantir que
    as lesões de margem entrem na folha.

    Retorna (fill_cor, borda_cor, maior_contorno):
      fill_cor  - máscara preenchida do maior contorno de cor
      borda_cor - contorno externo desenhado (linha), usado para fechar o Canny
      maior     - o maior contorno (ou None)
    """
    suave = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    lab = cv2.cvtColor(suave, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    _, verde = cv2.threshold(lab[:, :, 1], 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    marrom = _mascara_marrom(hsv)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    uni = cv2.bitwise_or(verde, marrom)
    uni = cv2.morphologyEx(uni, cv2.MORPH_CLOSE, k, iterations=4)
    uni = cv2.morphologyEx(uni, cv2.MORPH_OPEN, k, iterations=2)

    contornos, _ = cv2.findContours(uni, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill_cor = np.zeros(uni.shape, dtype=np.uint8)
    borda_cor = np.zeros(uni.shape, dtype=np.uint8)
    maior = None
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        cv2.drawContours(fill_cor, [maior], -1, 255, cv2.FILLED)
        cv2.drawContours(borda_cor, [maior], -1, 255, 2)
    return fill_cor, borda_cor, maior


def _maior_componente(mascara):
    """Mantém apenas o maior componente conectado da máscara."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n <= 1:
        return mascara
    idx = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]))
    return ((labels == idx) * 255).astype(np.uint8)


def _remover_peciolo(mascara, ks=25):
    """
    Remove o pecíolo (cabinho) SEM arredondar as serrilhas/dentes da folha.

    Em vez de erodir/dilatar a máscara inteira (o que cortaria os dentes), isola
    as PROTUBERÂNCIAS = máscara − abertura(máscara). Dentes e pecíolo viram
    protuberâncias; o pecíolo é a protuberância longa/grande na BASE da folha.
    Removemos só essa, preservando os dentes em todo o resto do contorno.
    """
    opened = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)), 1)
    protr = cv2.subtract(mascara, opened)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(protr)
    out = mascara.copy()
    Himg = mascara.shape[0]
    for i in range(1, n):
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        A = stats[i, cv2.CC_STAT_AREA]
        na_base = (y + h) > 0.82 * Himg
        alongado_ou_grande = (max(w, h) / max(1, min(w, h)) > 1.6) or (A > 0.004 * mascara.size)
        if na_base and alongado_ou_grande:
            out[labels == i] = 0
    return _maior_componente(out)


def _mascara_folha_canny(img_rgb, ks_peciolo=25):
    """
    MÁSCARA DA FOLHA (APENAS a folha, borda JUSTA) com o CANNY detectando a borda.

    Correções de aperto (resultado anterior estava "inchado" e pegava fundo):
      - morfologia MÍNIMA (kernels 3×3), para não arredondar as serrilhas
      - Canny detecta a borda real e o contorno é grampeado justo à cor
      - remove o fundo/sombra acromático (cinza) que entrava perto da base
      - remove o pecíolo preservando os dentes (ver _remover_peciolo)

    Etapas:
      1. Pré-máscara de cor (verde LAB ⋃ marrom HSV), morfologia 3×3 -> contorno justo
      2. Canny no canal L dentro da ROI -> borda; fecha leve -> preenche
      3. Grampeia justo à cor (dilatação 3×3)
      4. Remove acromático (cinza) + pecíolo -> APENAS a folha

    Retorna (mascara_uint8, maior_contorno).
    """
    suave = cv2.bilateralFilter(img_rgb, 7, 60, 60)
    lab = cv2.cvtColor(suave, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    _, verde = cv2.threshold(lab[:, :, 1], 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    marrom = _mascara_marrom(hsv)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    uni = cv2.morphologyEx(cv2.bitwise_or(verde, marrom), cv2.MORPH_OPEN, k3, 1)
    uni = cv2.morphologyEx(uni, cv2.MORPH_CLOSE, k3, 1)
    c, _ = cv2.findContours(uni, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not c:
        return uni, None
    maior_cor = max(c, key=cv2.contourArea)
    tight = np.zeros(uni.shape, dtype=np.uint8)
    cv2.drawContours(tight, [maior_cor], -1, 255, cv2.FILLED)

    # 2. CANNY detecta a borda dentro de uma ROI estreita (morfologia mínima)
    roi = cv2.dilate(tight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    L = cv2.bitwise_and(lab[:, :, 0], lab[:, :, 0], mask=roi)
    edges = cv2.bitwise_and(_auto_canny(cv2.bilateralFilter(L, 7, 60, 60)), roi)
    borda_cor = np.zeros(uni.shape, dtype=np.uint8)
    cv2.drawContours(borda_cor, [maior_cor], -1, 255, 1)
    fronteira = cv2.morphologyEx(cv2.bitwise_or(edges, borda_cor), cv2.MORPH_CLOSE, k3, 1)
    c2, _ = cv2.findContours(fronteira, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mascara = np.zeros(uni.shape, dtype=np.uint8)
    if c2:
        cv2.drawContours(mascara, [max(c2, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    mascara = cv2.bitwise_and(mascara, cv2.dilate(tight, k3, 1))  # grampeia justo

    # 3. remove fundo/sombra acromático (cinza) que entrou perto da base
    acrom = ((S < 35) & (V > 70)).astype(np.uint8) * 255
    mascara = cv2.bitwise_and(mascara, cv2.bitwise_not(acrom))

    # 4. remove o pecíolo (preserva dentes) + maior componente + CLOSE mínimo
    mascara = _remover_peciolo(mascara, ks_peciolo)
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)

    cnts2, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maior = max(cnts2, key=cv2.contourArea) if cnts2 else None
    return mascara, maior
    return mascara, maior


# ─── Detecção de Bordas ──────────────────────────────────────────────────────

def detectar_bordas_canny(img, threshold1=None, threshold2=None,
                          mascara_folha=None, somente_borda_externa=True):
    """
    Detecção de bordas com Canny — versão robusta para folha sobre fundo.

    O fundo texturizado é ELIMINADO antes do Canny pela máscara de cor; a borda
    externa sai garantidamente fechada (vem do maior contorno).

    Parâmetros:
      threshold1/threshold2 : limiares manuais. Se None, automáticos (±33% da mediana).
      mascara_folha         : máscara da folha pronta (opcional). Se None, calculada
                              via _mascara_folha_canny (inclui lesões de margem).
      somente_borda_externa : True -> só o contorno externo limpo.
                              False -> contorno externo + nervuras/detalhes internos.
    """
    if len(img.shape) == 3:
        if mascara_folha is None:
            mascara_folha, maior = _mascara_folha_canny(img)
        else:
            cnts, _ = cv2.findContours(mascara_folha, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            maior = max(cnts, key=cv2.contourArea) if cnts else None

        borda_externa = np.zeros(mascara_folha.shape, dtype=np.uint8)
        if maior is not None:
            cv2.drawContours(borda_externa, [maior], -1, 255, 2)
        if somente_borda_externa:
            return borda_externa

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mascara_folha)
    else:
        mascara_folha = None
        borda_externa = None
        gray = img.copy()

    # Bilateral preserva melhor a borda real que o Gaussian + reduz textura
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    if threshold1 is None or threshold2 is None:
        amostra = blurred[blurred > 0] if mascara_folha is not None else blurred
        med = np.median(amostra) if amostra.size else 0
        threshold1 = int(max(0,   (1.0 - 0.33) * med))
        threshold2 = int(min(255, (1.0 + 0.33) * med))

    edges = cv2.Canny(image=blurred, threshold1=threshold1, threshold2=threshold2)

    if mascara_folha is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mascara_folha)
        edges = cv2.bitwise_or(edges, borda_externa)  # garante contorno externo fechado
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
    Isola a folha COMPLETA (tecido verde + manchas) do fundo, com o CANNY
    detectando a borda.

    Pipeline (ver _mascara_folha_canny):
      1. Pré-máscara de cor (verde LAB ⋃ marrom HSV) suprime o fundo
      2. CANNY detecta a borda real da folha dentro dessa região
      3. Borda fechada -> maior contorno -> preenchimento = máscara
      4. Grampeada à extensão de cor (cola nas serrilhas) + CLOSE final

    Por que unir o marrom: as lesões na margem não são verdes; sem elas na
    pré-máscara, o contorno as cortaria fora da folha.

    Retorna:
        folha_segmentada - imagem RGB com fundo removido (preto)
        mascara          - máscara binária da folha completa (uint8)
    """
    mascara, _ = _mascara_folha_canny(img_rgb)
    folha_segmentada = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara)
    return folha_segmentada, mascara


def detectar_lesoes_hsv(img_rgb, sensibilidade='uva', mascara_folha=None,
                        area_minima=45, max_aspecto=4.0, erosao_borda=13):
    """
    Segmenta manchas de doença (tons marrom, amarelo, bege) usando HSV.

    Parâmetros:
        img_rgb       - imagem RGB da folha
        sensibilidade - 'baixa', 'media', 'alta', 'grape_black_rot' ou 'uva'
                        ('uva' = faixa marrom calibrada nesta amostra)
        mascara_folha - máscara da folha (retorno de segmentar_folha_verde)
        area_minima   - área mínima (px) de um componente para virar lesão
        max_aspecto   - razão máxima max(w,h)/min(w,h); elimina formas alongadas
        erosao_borda  - erode a máscara da folha antes de procurar lesão, para
                        descartar a borda amarelada que encosta no fundo
                        (principal fonte de falso positivo). 0 = desliga.

    Retorna:
        mascara_lesao  - máscara binária com as regiões de lesão (uint8)
        img_lesao      - imagem RGB com apenas as regiões de lesão visíveis
        n_lesoes       - número de lesões detectadas
        bboxes         - lista de (x, y, w, h) de cada lesão
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    faixas = {
        'baixa': [
            (np.array([10, 60, 40]),  np.array([25, 255, 200])),
            (np.array([20, 50, 150]), np.array([35, 255, 255])),
        ],
        'media': [
            (np.array([5,  40,  30]), np.array([30, 255, 220])),
            (np.array([15, 30, 100]), np.array([40, 255, 255])),
            (np.array([0,  20,  60]), np.array([12, 200, 200])),
        ],
        'alta': [
            (np.array([0,  20,  20]), np.array([40, 255, 255])),
            (np.array([0,  10,  30]), np.array([15, 180, 180])),
        ],
        'grape_black_rot': [
            (np.array([0,   0,  0]),  np.array([179, 100,  45])),
            (np.array([0,  20,  0]),  np.array([ 20, 255,  90])),
            (np.array([0,  40, 30]),  np.array([ 20, 255, 160])),
            (np.array([20, 80, 30]),  np.array([ 35, 255, 110])),
        ],
        # Calibrado nos pixels reais desta amostra (marrom escuro saturado)
        'uva': [
            (np.array([0, 45, 25]),   np.array([22, 255, 175])),
        ],
    }

    combinada = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in faixas.get(sensibilidade, faixas['media']):
        combinada = cv2.bitwise_or(combinada, cv2.inRange(hsv, lower, upper))

    # Morfologia para remover ruído e conectar manchas próximas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combinada = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel, iterations=1)
    combinada = cv2.morphologyEx(
        combinada, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2)

    # Restringe ao INTERIOR da folha — erode a borda amarela que encosta no fundo
    if mascara_folha is not None:
        interior = mascara_folha
        if erosao_borda > 0:
            ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosao_borda, erosao_borda))
            interior = cv2.erode(mascara_folha, ke, iterations=1)
        combinada = cv2.bitwise_and(combinada, interior)

    # Componentes conectados — filtra por área e razão de aspecto
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combinada)
    bboxes = []
    mascara_final = np.zeros_like(combinada)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_minima:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / max(1, min(w, h)) > max_aspecto:
            continue
        mascara_final[labels == i] = 255
        bboxes.append((stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], w, h))

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
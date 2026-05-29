# Capstone — Detecção de Doenças em Plantas (PDI)

Projeto de Processamento Digital de Imagens utilizando o dataset **Plant Disease Detection** (Roboflow, formato YOLOv8 com bounding boxes por lesão).

| Bloco | Técnica |
| --- | --- |
| 1 | Threshold & Blur |
| 2 | Edge & Corner Detection (Canny, Sobel, Harris, Shi-Tomasi) |
| 3 | Detecção por ROI + Segmentação HSV + Blend/Paste |
| 4 | YOLOv8 treinado para localizar e classificar lesões |

---

## Pré-requisitos

- Python 3.10 ou superior
- Git

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd PID_Project
```

### 2. Crie e ative o ambiente virtual

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Registre o kernel do Jupyter

```bash
python -m ipykernel install --user --name capstone-pdi --display-name "Python (capstone-pdi)"
```

---

## API Key do Roboflow

O notebook baixa o dataset automaticamente na primeira execução usando a API do Roboflow.

1. Crie uma conta gratuita em [roboflow.com](https://roboflow.com)
2. Acesse **Settings → API** e copie sua *Private API Key*
3. No notebook, na célula da seção **1 · Dataset**, substitua:

```python
ROBOFLOW_API_KEY = "SUA_API_KEY_AQUI"
```

pela sua chave real. O dataset será baixado em `data/` automaticamente.

> Se o dataset já estiver presente em `data/` (pastas `train/`, `valid/`, `test/` e arquivo `data.yaml`), o download é pulado.

---

## Estrutura do projeto

```text
PID_Project/
├── data/                  # Dataset (gerado pelo Roboflow no primeiro run)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── models/                # Pesos YOLOv8 (gerados no treinamento)
├── notebook/
│   └── capstone_plant_disease_detection.ipynb
├── results/               # Imagens geradas pelo notebook
├── src/
│   ├── preprocessing.py
│   ├── detection.py
│   ├── yolo_detection.py
│   └── utils.py
└── requirements.txt
```

---

## Observações

- O treinamento do YOLOv8 (Bloco 4) leva ~20–40 min com GPU e várias horas sem GPU. Se o arquivo `models/plant_disease_best.pt` já existir, o treinamento é pulado.
- Todos os resultados visuais são salvos automaticamente em `results/`.

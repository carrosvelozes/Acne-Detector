# Sistema de Detecção de Acnes usando YOLOv8 com Anotação Manual

## Descrição do Problema

Este projeto implementa um sistema completo de detecção automática de acnes em imagens faciais utilizando deep learning. O sistema emprega a arquitetura YOLOv8 para detecção de objetos, adaptada especificamente para identificar e localizar lesões de acne através de bounding boxes.

O problema abordado consiste na detecção automática de acnes em um dataset de 689 imagens faciais organizadas em diretórios numerados. Devido à ausência de anotações pré-existentes, foi desenvolvida uma estratégia de anotação manual seguida de treinamento supervisionado.

## Metodologia

### Estratégia de Anotação

A estratégia de anotação manual foi escolhida pelos seguintes motivos técnicos:

1. **Precisão**: Anotação humana garante maior precisão na identificação de lesões reais versus artefatos visuais
2. **Controle de qualidade**: Permite estabelecer critérios consistentes para definição de acne detectável
3. **Flexibilidade**: Possibilita ajuste de critérios durante o processo de anotação
4. **Base confiável**: Cria ground truth de alta qualidade para treinamento supervisionado

**Critérios de anotação estabelecidos:**
- Lesões inflamatórias visíveis (pápulas, pústulas)
- Comedões fechados e abertos quando claramente definidos
- Inclusão da área inflamada circundante
- Exclusão de manchas pós-inflamatórias sem elevação
- Bounding boxes ajustados ao tamanho real da lesão

### Arquitetura do Sistema

O sistema implementa uma pipeline de três estágios:

1. **Anotação Manual**: Interface gráfica para marcação de bounding boxes
2. **Treinamento**: Fine-tuning do YOLOv8n com transfer learning
3. **Inferência**: Aplicação do modelo treinado ao dataset completo

### Modelo Base

**YOLOv8 Nano (yolov8n.pt)**
- Arquitetura: CSPDarknet53 backbone + PANet neck + YOLOv8 head
- Parâmetros: ~3M parâmetros
- Classe única: 'acne'
- Input size: 640x640 pixels
- Justificativa: Balanceamento entre velocidade e precisão para detecção em tempo real

### Pré-processamento

**Pipeline de normalização implementado:**
```python
- Redimensionamento inteligente (máximo 800px, mantendo aspect ratio)
- Normalização automática via YOLOv8 (0-1 scaling)
- Data augmentation durante treinamento:
  - Horizontal flip (probabilidade 0.3)
  - Rotação (-5° a +5°)
  - Ajustes de escala (0.8-1.2)
  - Translação (-5% a +5%)
  - Ajustes de brilho e contraste
```

### Divisão do Dataset

**Estratégia de divisão controlada:**
- Training: 70% das imagens anotadas
- Validation: 15% das imagens anotadas  
- Test: 15% das imagens anotadas
- Seed fixo (42) para reprodutibilidade
- Shuffle aleatório antes da divisão
- Verificação de não-sobreposição entre conjuntos

### Hiperparâmetros de Treinamento

```yaml
epochs: 100-150
batch_size: 8 (otimizado para datasets pequenos)
learning_rate: 0.001 (Adam com decay)
patience: 50 (early stopping)
confidence_threshold_training: 0.001
iou_threshold: 0.5
weight_decay: 0.0001
warmup_epochs: 1
optimizer: AdamW
```

## Instalação e Dependências

### Requisitos do Sistema
- Python 3.8+
- GPU CUDA opcional (CPU suportado)
- 4GB RAM mínimo
- 2GB espaço em disco

### Instalação
```bash
git clone <repository>
cd acne-detection
pip install -r requirements.txt
```

### Dependências Principais
```
ultralytics==8.0.196  # YOLOv8 framework
torch==2.0.1          # PyTorch backend
opencv-python==4.8.1.78  # Processamento de imagem
albumentations==1.3.1     # Data augmentation
pandas==2.0.3             # Manipulação de dados
matplotlib==3.7.2         # Visualização
```

## Estrutura do Projeto

```
acne-detection/
├── images/                          # Dataset de imagens
├── df.csv                          # Lista de caminhos das imagens
├── manual_annotation_tool.py       # Interface de anotação
├── train_manual_model.py          # Módulo de treinamento
├── apply_model_to_images.py       # Módulo de inferência
├── complete_acne_detection_pipeline.py  # Pipeline completo
├── requirements.txt                # Dependências
└── README.md                      # Documentação
```

## Instruções de Uso

### Pipeline Completo
```bash
# Execução completa: anotação + treinamento + inferência
python complete_acne_detection_pipeline.py --annotate 30 --epochs 100 --confidence 0.1

# Parâmetros opcionais:
# --annotate N: número de imagens para anotar
# --epochs N: épocas de treinamento
# --confidence F: threshold de confiança para detecção
# --skip-annotation: usar anotações existentes
# --skip-training: usar modelo existente
```

### Execução Modular
```bash
# 1. Anotação manual
python manual_annotation_tool.py

# 2. Treinamento do modelo
python train_manual_model.py

# 3. Aplicação às imagens
python apply_model_to_images.py
```

### Interface de Anotação
**Controles:**
- Click + drag: criar bounding box
- Right click: remover último box
- ENTER: salvar e próxima imagem
- ESC: salvar progresso e sair
- S: pular imagem atual

## Métricas de Avaliação

### Métricas Implementadas

**Para detecção de objetos:**
1. **mAP@0.5**: Mean Average Precision com IoU threshold 0.5
2. **mAP@0.5:0.95**: mAP médio para IoU de 0.5 a 0.95 (incrementos 0.05)
3. **Precision**: TP / (TP + FP)
4. **Recall**: TP / (TP + FN)
5. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

**Métricas de performance do sistema:**
- Taxa de detecção: % de imagens com pelo menos uma detecção
- Média de detecções por imagem
- Distribuição de scores de confiança
- Tempo médio de inferência por imagem

### Formato de Saída das Métricas

```json
{
  "mAP50": 0.65,
  "mAP50-95": 0.45,
  "precision": 0.78,
  "recall": 0.82,
  "f1_score": 0.80,
  "detection_rate": 0.73,
  "avg_detections_per_image": 2.3
}
```

## Resultados e Arquivos Gerados

### Estrutura de Output

```
manual_annotations/
├── annotations.json              # Anotações originais
└── yolo_dataset/                # Dataset formatado
    ├── train/                   # Conjunto de treinamento
    ├── val/                     # Conjunto de validação
    ├── test/                    # Conjunto de teste
    └── acne_config.yaml         # Configuração YOLO

trained_model/
├── acne_model/weights/
│   ├── best.pt                  # Melhor modelo (menor validation loss)
│   └── last.pt                  # Último checkpoint
├── training_info.json           # Métricas de treinamento
└── evaluation_metrics.json     # Métricas de avaliação

detection_results/
├── images_with_detections/      # Imagens com bounding boxes
├── images_without_detections/   # Imagens sem detecções
├── detection_crops/             # Crops individuais das lesões
├── detection_results.json      # Resultados estruturados
├── detection_summary.png       # Gráficos de análise
└── detection_report.html       # Relatório completo
```

### Formato dos Resultados

**detection_results.json:**
```json
{
  "timestamp": "2024-XX-XX",
  "model_path": "trained_model/acne_model/weights/best.pt",
  "conf_threshold": 0.1,
  "total_images_processed": 689,
  "images_with_detections": 503,
  "total_detections": 1247,
  "detection_rate": 0.73,
  "results": [
    {
      "image_path": "/1/1.jpg",
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.85,
          "class": 0
        }
      ]
    }
  ]
}
```

## Configurações Avançadas

### Personalização de Treinamento

**Arquivo train_manual_model.py - Hiperparâmetros:**
```python
epochs = 100           # Número de épocas
batch_size = 8         # Tamanho do batch
lr0 = 0.001           # Learning rate inicial
patience = 50         # Early stopping patience
conf_threshold = 0.001 # Confidence para treinamento
iou_threshold = 0.5   # IoU para NMS
```

### Personalização de Detecção

**Arquivo apply_model_to_images.py - Parâmetros:**
```python
conf_threshold = 0.1   # Threshold de confiança mínima
iou_threshold = 0.7    # IoU para Non-Maximum Suppression
max_det = 300         # Máximo de detecções por imagem
```

## Limitações e Considerações

### Limitações Técnicas
1. **Dataset size**: Performance dependente do número de anotações manuais
2. **Generalização**: Modelo específico para o tipo de imagens do dataset
3. **Resolução**: Otimizado para imagens de até 800px
4. **Classe única**: Não diferencia tipos de lesões de acne

### Considerações de Performance
- Mínimo 30 imagens anotadas para treinamento básico
- Recomendado 100+ imagens para performance robusta
- GPU acelera treinamento mas não é obrigatória
- Tempo de treinamento: ~10-30 minutos (CPU), ~5-10 minutos (GPU)

## Troubleshooting

### Problemas Comuns

**"Modelo não encontrado"**
```bash
# Executar treinamento primeiro
python train_manual_model.py
```

**"Dataset não encontrado"**
```bash
# Executar anotação primeiro
python manual_annotation_tool.py
```

**Low performance / métricas baixas**
- Aumentar número de imagens anotadas
- Verificar qualidade das anotações
- Ajustar threshold de confiança
- Aumentar épocas de treinamento

**Erro de memória durante treinamento**
- Reduzir batch_size no arquivo train_manual_model.py
- Usar CPU ao invés de GPU
- Reduzir resolução das imagens

## Reprodutibilidade

Para garantir reprodutibilidade dos resultados:

1. **Seeds fixos**: seed=42 em todas as operações aleatórias
2. **Versões específicas**: Dependências com versões fixas
3. **Configurações documentadas**: Todos os hiperparâmetros especificados
4. **Divisão determinística**: Mesmo conjunto train/val/test sempre

**Comando para reprodução exata:**
```bash
python complete_acne_detection_pipeline.py --annotate 50 --epochs 100 --confidence 0.1
```

## Referências Técnicas

- **YOLOv8**: Ultralytics YOLOv8 Documentation
- **Transfer Learning**: Fine-tuning de modelos pré-treinados
- **Data Augmentation**: Albumentations library
- **Evaluation Metrics**: COCO evaluation protocol
- **PyTorch**: Deep learning framework

## Detalhes da Arquitetura e Decisões Técnicas

### Escolha da Arquitetura YOLOv8

**Justificativa para YOLOv8 Nano:**
1. **Eficiência computacional**: Adequado para recursos limitados
2. **Velocidade de inferência**: ~10-50ms por imagem em CPU
3. **Tamanho do modelo**: 6MB, adequado para deployment
4. **Performance balanceada**: Boa precisão versus velocidade
5. **Transfer learning**: Pré-treinado em COCO dataset com 80 classes

**Arquitetura detalhada:**
```
Backbone: CSPDarknet53
- 53 camadas convolucionais
- Cross Stage Partial connections
- Spatial Pyramid Pooling (SPP)

Neck: Path Aggregation Network (PANet)
- Feature Pyramid Network (FPN) + bottom-up augmentation
- Múltiplas escalas de feature maps

Head: YOLOv8 Detection Head
- Anchor-free detection
- Decoupled classification e regression heads
- Distribution Focal Loss para bounding box regression
```

### Pipeline de Pré-processamento Técnico

**Estágio 1: Validação e Carregamento**
```python
# Verificação de integridade da imagem
if len(image.shape) != 3:
    reject_image()

# Logging de dimensões originais
original_shape = image.shape  # [H, W, C]
```

**Estágio 2: Redimensionamento Adaptativo**
```python
# Manter aspect ratio, limitar tamanho máximo
if max(h, w) > max_size:
    scale = max_size / max(h, w)
    new_dimensions = (int(w * scale), int(h * scale))
    # Interpolação Lanczos para preservar detalhes
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LANCZOS4)
```

**Estágio 3: Normalização (YOLOv8 interno)**
```python
# Aplicado automaticamente pelo modelo:
# 1. Conversão RGB para tensor PyTorch
# 2. Normalização 0-255 -> 0-1
# 3. Padding para múltiplos de 32 (grid size)
```

### Estratégia de Transfer Learning

**Base model initialization:**
- YOLOv8n pré-treinado no COCO dataset
- Freezing das primeiras camadas do backbone
- Fine-tuning apenas das últimas camadas + detection head

**Adaptação para acne detection:**
```yaml
# Mudança de 80 classes COCO para 1 classe
nc: 1  # number of classes
names: ['acne']  # class names

# Ajuste da confidence threshold
conf: 0.001  # para training
conf: 0.1    # para inference
```

### Otimizações para Dataset Pequeno

**Hiperparâmetros ajustados:**
```python
batch_size: 8           # Reduzido para evitar overfitting
learning_rate: 0.001    # Menor para convergência estável  
weight_decay: 0.0001    # Regularização leve
dropout: 0.1            # Prevenção de overfitting
warmup_epochs: 1        # Warmup mínimo
patience: 50            # Early stopping generoso
```

**Data Augmentation reduzida:**
```python
# Augmentations conservadoras para dataset pequeno
mosaic: 0.8        # Reduzido de 1.0
mixup: 0.0         # Desabilitado
degrees: 5.0       # Rotação mínima ±5°
translate: 0.05    # Translação mínima ±5%
scale: 0.2         # Escala mínima ±20%
fliplr: 0.3        # Flip horizontal reduzido
```

### Estratégia de Avaliação

**Métricas implementadas:**
1. **mAP@0.5**: Métrica principal para object detection
2. **mAP@0.5:0.95**: Métrica mais rigorosa, multiple IoU thresholds
3. **Precision**: Quantas detecções estão corretas
4. **Recall**: Quantas acnes reais foram detectadas
5. **F1-Score**: Harmonic mean de precision e recall

**Interpretação das métricas:**
```
mAP@0.5 > 0.5     : Modelo aceitável
mAP@0.5 > 0.7     : Modelo bom
mAP@0.5 > 0.8     : Modelo excelente

Precision > 0.8   : Poucas detecções falsas
Recall > 0.8      : Detecta maioria das acnes reais
F1-Score > 0.8    : Balanceamento bom entre P e R
```

### Decisões de Implementation

**1. Interface de anotação personalizada vs. ferramentas existentes:**
- **Escolha**: Interface customizada em OpenCV
- **Justificativa**: Controle total do processo, integração direta com pipeline
- **Trade-off**: Mais desenvolvimento vs. flexibilidade máxima

**2. YOLOv8 vs. outras arquiteturas:**
- **Alternatives consideradas**: Faster R-CNN, RetinaNet, EfficientDet
- **Escolha**: YOLOv8 pela velocidade e facilidade de deployment
- **Trade-off**: Velocidade vs. precisão máxima

**3. Dataset split strategy:**
- **Escolha**: 70/15/15 split com shuffle aleatório
- **Justificativa**: Padrão da literatura, balanceamento adequado
- **Consideração**: Verificação manual de não-overlap entre splits

**4. Confidence threshold para inference:**
- **Escolha**: 0.1 (baixo threshold)
- **Justificativa**: Preferir detectar mais acnes (high recall) vs. precisão máxima
- **Configurável**: Usuário pode ajustar via parâmetro

### Considerações de Escalabilidade

**Para datasets maiores (1000+ imagens):**
```python
# Ajustes recomendados:
batch_size: 16-32
learning_rate: 0.01
epochs: 200-300
data_augmentation: full pipeline
```

**Para deployment em produção:**
```python
# Otimizações:
model_export: ONNX or TensorRT
quantization: INT8 for speed
batch_inference: multiple images per call
caching: pre-computed features
```

### Limitações Técnicas Identificadas

**1. Dataset dependency:**
- Performance diretamente correlacionada com qualidade e quantidade de anotações
- Minimum viable: 30 imagens, recommended: 100+

**2. Domain specificity:**
- Modelo treinado especificamente para o tipo de imagens do dataset
- Generalização limitada para outros tipos de fotografia facial

**3. Single-class limitation:**
- Não diferencia tipos de acne (comedão, pápula, pústula)
- Extensão para multi-class requer re-anotação do dataset

**4. Resolution constraints:**
- Otimizado para imagens de até 800-1280px
- Performance pode degradar em resoluções muito altas ou baixas 
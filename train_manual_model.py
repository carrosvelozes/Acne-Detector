import os
import torch
from ultralytics import YOLO
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import cv2
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManualModelTrainer:
    def __init__(self, config_path: str, output_dir: str = "trained_model"):
        """
        Treinador YOLOv8 para detecção de acnes usando anotações manuais
        
        Args:
            config_path: Caminho para arquivo de configuração YOLO
            output_dir: Diretório de output
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Carregar configuração
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Verificar se CUDA está disponível
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Usando device: {self.device}")
        
        # Modelo treinado
        self.trained_model = None
        
    def train_model(self, 
                   epochs: int = 100,
                   imgsz: int = 640,
                   batch_size: int = 8,  # Reduzido para dataset pequeno
                   lr0: float = 0.001,   # Learning rate menor
                   patience: int = 50):
        """
        Treinar modelo YOLOv8 com anotações manuais
        
        Args:
            epochs: Número de épocas
            imgsz: Tamanho da imagem
            batch_size: Tamanho do batch
            lr0: Learning rate inicial
            patience: Paciência para early stopping
        """
        logger.info("Iniciando treinamento do YOLOv8 com anotações manuais...")
        
        # Inicializar modelo YOLOv8
        model = YOLO('yolov8n.pt')  # YOLOv8 nano
        
        # Configurações de treinamento otimizadas para dataset pequeno
        train_args = {
            'data': str(self.config_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'lr0': lr0,
            'patience': patience,
            'project': str(self.output_dir),
            'name': 'acne_model',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': True,
            'cos_lr': True,
            'close_mosaic': 5,    # Reduzido para dataset pequeno
            'resume': False,
            'amp': False,         # Desabilitado para estabilidade
            'fraction': 1.0,
            'dropout': 0.1,       # Pequeno dropout para evitar overfitting
            'val': True,
            'plots': True,
            'save': True,
            'save_json': True,
            'conf': 0.001,        # Threshold muito baixo para treino
            'iou': 0.5,           # IOU mais baixo
            'max_det': 100,
            'half': False,
            'dnn': False,
            'weight_decay': 0.0001,  # Regularização menor
            'warmup_epochs': 1,      # Menos warmup para dataset pequeno
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'mosaic': 0.8,           # Menos mosaic augmentation
            'mixup': 0.0,            # Sem mixup para dataset pequeno
            'copy_paste': 0.0,       # Sem copy-paste
            'degrees': 5.0,          # Rotação mínima
            'translate': 0.05,       # Translação mínima
            'scale': 0.2,            # Escala mínima
            'shear': 2.0,            # Shear mínimo
            'flipud': 0.0,           # Sem flip vertical
            'fliplr': 0.3,           # Flip horizontal reduzido
        }
        
        try:
            # Treinar modelo
            logger.info(f"Iniciando treinamento com {epochs} épocas...")
            results = model.train(**train_args)
            
            # Salvar informações do treinamento
            training_info = {
                'epochs_completed': results.epoch + 1 if hasattr(results, 'epoch') else epochs,
                'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
                'final_metrics': {},
                'training_args': train_args
            }
            
            # Extrair métricas finais se disponíveis
            if hasattr(results, 'results_dict'):
                training_info['final_metrics'] = {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                    'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
                }
            
            # Salvar informações
            with open(self.output_dir / "training_info.json", 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info("Treinamento concluído com sucesso!")
            logger.info(f"Métricas finais: {training_info['final_metrics']}")
            
            # Carregar melhor modelo
            best_model_path = self.output_dir / "acne_model" / "weights" / "best.pt"
            if best_model_path.exists():
                self.trained_model = YOLO(str(best_model_path))
                logger.info(f"Melhor modelo carregado: {best_model_path}")
            else:
                self.trained_model = model
                logger.warning("Melhor modelo não encontrado, usando modelo atual")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {e}")
            raise e
    
    def evaluate_model(self):
        """Avaliar modelo no conjunto de teste"""
        
        if self.trained_model is None:
            logger.error("Modelo não foi treinado ainda!")
            return None
        
        logger.info("Avaliando modelo...")
        
        try:
            # Avaliar modelo
            results = self.trained_model.val(
                data=str(self.config_path),
                split='test',
                imgsz=640,
                batch=16,
                conf=0.25,
                iou=0.7,
                max_det=300,
                device=self.device,
                plots=True,
                save_json=True,
                project=str(self.output_dir),
                name='evaluation',
                exist_ok=True
            )
            
            # Extrair métricas
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
            }
            
            logger.info("Métricas de avaliação:")
            for metric, value in metrics.items():
                logger.info(f"- {metric}: {value:.4f}")
            
            # Salvar métricas
            with open(self.output_dir / "evaluation_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro durante avaliação: {e}")
            return None
    
    def get_model_path(self):
        """Retornar caminho do melhor modelo treinado"""
        best_model_path = self.output_dir / "acne_model" / "weights" / "best.pt"
        if best_model_path.exists():
            return str(best_model_path)
        return None


def main():
    """Função principal para treinar modelo com anotações manuais"""
    
    # Configurações
    CONFIG_PATH = "manual_annotations/yolo_dataset/acne_config.yaml"
    OUTPUT_DIR = "trained_model"
    
    # Verificar se o arquivo de configuração existe
    if not Path(CONFIG_PATH).exists():
        logger.error(f"Arquivo de configuração não encontrado: {CONFIG_PATH}")
        logger.error("Execute primeiro manual_annotation_tool.py para criar anotações")
        return None
    
    # Verificar se existem dados de treino
    dataset_dir = Path("manual_annotations/yolo_dataset")
    train_images = dataset_dir / "train" / "images"
    if not train_images.exists() or not any(train_images.iterdir()):
        logger.error("Dados de treino não encontrados!")
        logger.error("Execute manual_annotation_tool.py primeiro")
        return None
    
    print("=== TREINAMENTO DO MODELO COM ANOTAÇÕES MANUAIS ===")
    print(f"Configuração: {CONFIG_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Inicializar treinador
    trainer = ManualModelTrainer(CONFIG_PATH, OUTPUT_DIR)
    
    # Configurações de treinamento
    epochs = 150  # Mais épocas para anotações manuais
    batch_size = 16
    lr0 = 0.01
    patience = 50
    
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr0}")
    print(f"Device: {trainer.device}")
    print()
    
    input("Pressione ENTER para iniciar o treinamento...")
    
    # Treinar modelo
    try:
        training_results = trainer.train_model(
            epochs=epochs,
            batch_size=batch_size,
            lr0=lr0,
            patience=patience
        )
        
        # Avaliar modelo
        eval_metrics = trainer.evaluate_model()
        
        # Informações finais
        model_path = trainer.get_model_path()
        
        print("\n=== TREINAMENTO CONCLUÍDO ===")
        if eval_metrics:
            print("Métricas finais:")
            for metric, value in eval_metrics.items():
                print(f"- {metric}: {value:.4f}")
        
        if model_path:
            print(f"\nModelo salvo em: {model_path}")
            print("Próximo passo: executar apply_model_to_images.py")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        return None


if __name__ == "__main__":
    main() 
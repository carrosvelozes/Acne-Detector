#!/usr/bin/env python3
"""
Pipeline completo e correto para detecção de acnes:

1. ANOTAÇÃO MANUAL: Marcar bounding boxes nas espinhas
2. TREINAMENTO: Treinar YOLOv8 com as anotações manuais  
3. APLICAÇÃO: Usar modelo treinado para detectar acnes em todas as outras imagens
4. RESULTADOS: Salvar imagens com bounding boxes desenhados

Autor: Sistema de Detecção de Acnes
Data: 2024
"""

import os
import sys
import time
import logging
from pathlib import Path
import argparse

# Configurar logging para Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('acne_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Verificar se todos os arquivos necessários existem"""
    
    logger.info("Verificando requisitos...")
    
    required_files = [
        "images",  # Diretório de imagens
        "df.csv",  # Lista de imagens
        "manual_annotation_tool.py",
        "train_manual_model.py", 
        "apply_model_to_images.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"ERRO: Arquivos não encontrados: {missing_files}")
        return False
    
    # Verificar se o diretório de imagens não está vazio
    images_dir = Path("images")
    if not any(images_dir.iterdir()):
        logger.error("ERRO: Diretório de imagens está vazio")
        return False
    
    logger.info("OK: Todos os requisitos verificados")
    return True

def step1_manual_annotation(num_images_to_annotate: int = 50):
    """
    ETAPA 1: Anotação manual de bounding boxes
    
    Args:
        num_images_to_annotate: Número de imagens para anotar manualmente
    """
    logger.info("=== ETAPA 1: ANOTAÇÃO MANUAL ===")
    logger.info(f"Vamos anotar {num_images_to_annotate} imagens manualmente")
    
    try:
        from manual_annotation_tool import ManualAnnotationTool
        
        # Inicializar ferramenta
        tool = ManualAnnotationTool("images", "df.csv", "manual_annotations")
        
        print(f"\n🖱️  INSTRUÇÕES PARA ANOTAÇÃO:")
        print("- Clique e arraste para marcar uma espinha")
        print("- Botão direito: remove último bounding box")  
        print("- ENTER: salva e vai para próxima imagem")
        print("- ESC: sair e salvar progresso")
        print("- S: pular imagem atual")
        print()
        
        input("Pressione ENTER para começar a anotação manual...")
        
        # Fazer anotações manuais
        annotations = tool.annotate_images(
            start_index=0,
            max_images=num_images_to_annotate
        )
        
        if not annotations or len(annotations) == 0:
            logger.error("ERRO: Nenhuma anotação foi criada")
            return False
        
        # Criar dataset YOLO
        config_path = tool.create_yolo_dataset(annotations)
        
        if not config_path:
            logger.error("ERRO: Falha na criação do dataset YOLO")
            return False
        
        annotated_count = len([ann for ann in annotations if ann['num_acnes'] > 0])
        total_acnes = sum([ann['num_acnes'] for ann in annotations])
        
        logger.info(f"✅ Etapa 1 concluída:")
        logger.info(f"   - {len(annotations)} imagens processadas")
        logger.info(f"   - {annotated_count} imagens com anotações")
        logger.info(f"   - {total_acnes} acnes marcadas")
        logger.info(f"   - Dataset YOLO criado: {config_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"ERRO na anotação manual: {e}")
        return False

def step2_train_model(epochs: int = 150):
    """
    ETAPA 2: Treinar modelo YOLOv8 com anotações manuais
    
    Args:
        epochs: Número de épocas para treinamento
    """
    logger.info("=== ETAPA 2: TREINAMENTO DO MODELO ===")
    
    # Verificar se dataset existe
    config_path = "manual_annotations/yolo_dataset/acne_config.yaml"
    if not Path(config_path).exists():
        logger.error("ERRO: Dataset não encontrado. Execute a Etapa 1 primeiro.")
        return False
    
    try:
        from train_manual_model import ManualModelTrainer
        
        # Inicializar treinador
        trainer = ManualModelTrainer(config_path, "trained_model")
        
        logger.info(f"Iniciando treinamento com {epochs} épocas...")
        logger.info(f"Device: {trainer.device}")
        
        # Treinar modelo
        training_results = trainer.train_model(
            epochs=epochs,
            batch_size=16,
            lr0=0.01,
            patience=50
        )
        
        if not training_results:
            logger.error("ERRO: Falha no treinamento")
            return False
        
        # Avaliar modelo
        eval_metrics = trainer.evaluate_model()
        
        # Verificar se modelo foi salvo
        model_path = trainer.get_model_path()
        if not model_path:
            logger.error("ERRO: Modelo treinado não foi salvo corretamente")
            return False
        
        logger.info(f"✅ Etapa 2 concluída:")
        logger.info(f"   - Modelo salvo: {model_path}")
        if eval_metrics:
            logger.info(f"   - mAP@0.5: {eval_metrics.get('mAP50', 'N/A'):.4f}")
            logger.info(f"   - Precision: {eval_metrics.get('precision', 'N/A'):.4f}")
            logger.info(f"   - Recall: {eval_metrics.get('recall', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ERRO no treinamento: {e}")
        return False

def step3_apply_to_all_images(conf_threshold: float = 0.25):
    """
    ETAPA 3: Aplicar modelo treinado a todas as imagens
    
    Args:
        conf_threshold: Threshold de confiança para detecções
    """
    logger.info("=== ETAPA 3: APLICAÇÃO DO MODELO ===")
    
    # Verificar se modelo existe
    model_path = "trained_model/acne_model/weights/best.pt"
    if not Path(model_path).exists():
        logger.error("ERRO: Modelo treinado não encontrado. Execute a Etapa 2 primeiro.")
        return False
    
    try:
        from apply_model_to_images import AcneModelApplicator
        
        # Inicializar aplicador
        applicator = AcneModelApplicator(
            model_path=model_path,
            images_dir="images",
            csv_path="df.csv",
            output_dir="detection_results"
        )
        
        logger.info(f"Processando {len(applicator.image_paths)} imagens...")
        logger.info(f"Threshold de confiança: {conf_threshold}")
        
        # Processar todas as imagens
        results = applicator.process_all_images(
            conf_threshold=conf_threshold,
            max_images=None  # Processar TODAS as imagens
        )
        
        # Criar relatório
        applicator.create_summary_report(results)
        
        logger.info(f"✅ Etapa 3 concluída:")
        logger.info(f"   - {results['total_images_processed']} imagens processadas")
        logger.info(f"   - {results['images_with_detections']} imagens com acnes detectadas")
        logger.info(f"   - {results['total_detections']} acnes encontradas")
        logger.info(f"   - Taxa de detecção: {results['detection_rate']*100:.1f}%")
        logger.info(f"   - Relatório: detection_results/detection_report.html")
        
        return True
        
    except Exception as e:
        logger.error(f"ERRO na aplicação do modelo: {e}")
        return False

def print_final_summary():
    """Imprimir resumo final dos resultados"""
    
    logger.info("=== RESUMO FINAL ===")
    
    # Verificar arquivos gerados
    key_paths = {
        "Anotações manuais": "manual_annotations/annotations.json",
        "Dataset YOLO": "manual_annotations/yolo_dataset/acne_config.yaml", 
        "Modelo treinado": "trained_model/acne_model/weights/best.pt",
        "Imagens com detecções": "detection_results/images_with_detections/",
        "Imagens sem detecções": "detection_results/images_without_detections/",
        "Crops das acnes": "detection_results/detection_crops/",
        "Relatório final": "detection_results/detection_report.html"
    }
    
    logger.info("📁 Arquivos e diretórios gerados:")
    for description, path in key_paths.items():
        status = "✅" if Path(path).exists() else "❌"
        logger.info(f"   {status} {description}: {path}")
    
    # Carregar estatísticas finais se disponíveis
    results_file = Path("detection_results/detection_results.json")
    if results_file.exists():
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info("\n📊 Estatísticas finais:")
            logger.info(f"   - Total de imagens processadas: {results['total_images_processed']}")
            logger.info(f"   - Imagens com acnes: {results['images_with_detections']}")
            logger.info(f"   - Total de acnes detectadas: {results['total_detections']}")
            logger.info(f"   - Taxa de detecção: {results['detection_rate']*100:.1f}%")
            
        except Exception as e:
            logger.warning(f"Não foi possível carregar estatísticas: {e}")
    
    # Instruções finais
    report_path = Path("detection_results/detection_report.html")
    if report_path.exists():
        logger.info(f"\n🎯 MISSÃO CUMPRIDA!")
        logger.info(f"   Abra o relatório: {report_path.absolute()}")
        logger.info(f"   - Imagens com bounding boxes estão em: detection_results/images_with_detections/")
        logger.info(f"   - Crops individuais das acnes: detection_results/detection_crops/")

def main():
    """Função principal do pipeline"""
    parser = argparse.ArgumentParser(description='Pipeline completo de detecção de acnes')
    parser.add_argument('--annotate', type=int, default=30, 
                       help='Número de imagens para anotar manualmente (padrão: 30)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas para treinamento (padrão: 100)')
    parser.add_argument('--confidence', type=float, default=0.1,
                       help='Threshold de confiança para detecção (padrão: 0.1)')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Pular etapa de anotação (usar existente)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Pular etapa de treinamento (usar modelo existente)')
    
    args = parser.parse_args()
    
    # Banner inicial
    print("="*80)
    print("🔍 SISTEMA COMPLETO DE DETECÇÃO DE ACNES")
    print("="*80)
    print("ETAPAS:")
    print("1. 🖱️  Anotação manual de bounding boxes")
    print("2. 🤖 Treinamento do modelo YOLOv8")  
    print("3. 🎯 Aplicação nas outras imagens")
    print("4. 💾 Salvamento com bounding boxes desenhados")
    print("="*80)
    print(f"Configurações:")
    print(f"   - Imagens para anotar: {args.annotate}")
    print(f"   - Épocas de treinamento: {args.epochs}")
    print(f"   - Threshold de confiança: {args.confidence}")
    print("="*80)
    
    start_time = time.time()
    
    # Verificar requisitos
    if not check_requirements():
        logger.error("Pipeline interrompido devido a requisitos não atendidos")
        return 1
    
    # ETAPA 1: Anotação manual (opcional)
    if not args.skip_annotation:
        if not step1_manual_annotation(args.annotate):
            logger.error("Pipeline interrompido na etapa de anotação")
            return 1
    else:
        logger.info("Pulando etapa de anotação")
        if not Path("manual_annotations/yolo_dataset/acne_config.yaml").exists():
            logger.error("Dataset não encontrado! Execute anotação primeiro.")
            return 1
    
    # ETAPA 2: Treinamento (opcional)
    if not args.skip_training:
        if not step2_train_model(args.epochs):
            logger.error("Pipeline interrompido na etapa de treinamento")
            return 1
    else:
        logger.info("Pulando etapa de treinamento")
        if not Path("trained_model/acne_model/weights/best.pt").exists():
            logger.error("Modelo não encontrado! Execute treinamento primeiro.")
            return 1
    
    # ETAPA 3: Aplicação do modelo (obrigatória)
    if not step3_apply_to_all_images(args.confidence):
        logger.error("Pipeline interrompido na etapa de aplicação")
        return 1
    
    # Tempo total
    total_time = time.time() - start_time
    
    # Resumo final
    print("\n" + "="*80)
    print("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"⏱️  Tempo total: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
    print()
    
    print_final_summary()
    
    print("="*80)
    print("✨ Sistema de Detecção de Acnes finalizado!")
    print("   Todas as imagens foram processadas e salvas com bounding boxes.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        sys.exit(1) 
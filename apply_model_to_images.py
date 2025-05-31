import os
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AcneModelApplicator:
    def __init__(self, model_path: str, images_dir: str, csv_path: str, output_dir: str = "detection_results"):
        """
        Aplicador do modelo treinado para detecção de acnes
        
        Args:
            model_path: Caminho para modelo treinado (.pt)
            images_dir: Diretório das imagens
            csv_path: Arquivo CSV com lista de imagens
            output_dir: Diretório para salvar resultados
        """
        self.model_path = Path(model_path)
        self.images_dir = Path(images_dir)
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        
        # Criar diretórios de output
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images_with_detections").mkdir(exist_ok=True)
        (self.output_dir / "images_without_detections").mkdir(exist_ok=True)
        (self.output_dir / "detection_crops").mkdir(exist_ok=True)
        
        # Carregar modelo
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        logger.info(f"Carregando modelo: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Carregar lista de imagens
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df['files'].tolist()
        
        logger.info(f"Modelo carregado. {len(self.image_paths)} imagens para processar")
        
    def preprocess_image(self, image_path: str):
        """
        Pipeline explícito de pré-processamento de imagens
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            tuple: (imagem_processada, info_processamento) ou (None, None) se erro
        """
        full_path = self.images_dir / image_path.lstrip('/')
        
        if not full_path.exists():
            logger.error(f"Imagem não encontrada: {full_path}")
            return None, None
        
        # Carregar imagem
        image = cv2.imread(str(full_path))
        if image is None:
            logger.error(f"Erro ao carregar imagem: {full_path}")
            return None, None
        
        # Informações originais
        original_shape = image.shape
        processing_info = {
            'original_shape': original_shape,
            'original_size': original_shape[1] * original_shape[0],
            'channels': original_shape[2] if len(original_shape) == 3 else 1,
            'preprocessing_applied': []
        }
        
        # Verificação de qualidade
        if len(image.shape) != 3:
            logger.warning(f"Imagem não é RGB: {full_path}")
            return None, None
        
        # Redimensionamento se necessário (YOLOv8 faz isso internamente, mas documentamos)
        h, w = image.shape[:2]
        if max(h, w) > 1280:  # Limite para evitar problemas de memória
            scale = 1280 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            processing_info['preprocessing_applied'].append(f'resize_{scale:.3f}')
        
        # Normalização é feita internamente pelo YOLOv8 (0-255 -> 0-1)
        processing_info['preprocessing_applied'].append('yolo_normalization')
        
        return image, processing_info
    
    def detect_acnes_in_image(self, image_path: str, conf_threshold: float = 0.1):
        """
        Detectar acnes em uma imagem usando pipeline de pré-processamento explícito
        
        Args:
            image_path: Caminho da imagem
            conf_threshold: Threshold de confiança
            
        Returns:
            dict: Resultados da detecção
        """
        # Aplicar pipeline de pré-processamento
        image, processing_info = self.preprocess_image(image_path)
        
        if image is None:
            return None
        
        # Fazer predição com configurações explícitas
        results = self.model.predict(
            source=image,  # Usar imagem pré-processada
            conf=conf_threshold,
            iou=0.7,
            max_det=300,
            device='cpu',  # Usar CPU para consistência
            save=False,
            verbose=False,
            imgsz=640,  # Tamanho padrão YOLOv8
            augment=False,  # Sem augmentação durante inferência
            half=False,  # Precisão completa
            visualize=False  # Sem visualização automática
        )
        
        if len(results) == 0:
            return {
                'image_path': image_path,
                'detections': [],
                'num_detections': 0,
                'has_acne': False,
                'preprocessing_info': processing_info
            }
        
        result = results[0]
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class': int(box.cls[0].cpu().numpy())
                }
                detections.append(detection)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections),
            'has_acne': len(detections) > 0,
            'preprocessing_info': processing_info
        }
    
    def draw_detections_on_image(self, image_path: str, detections: list, save_path: str):
        """
        Desenhar detecções na imagem e salvar
        
        Args:
            image_path: Caminho da imagem original
            detections: Lista de detecções
            save_path: Caminho para salvar imagem com detecções
        """
        full_path = self.images_dir / image_path.lstrip('/')
        
        # Carregar imagem
        image = cv2.imread(str(full_path))
        if image is None:
            logger.error(f"Erro ao carregar imagem: {full_path}")
            return False
        
        # Desenhar cada detecção
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Desenhar retângulo
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adicionar texto com confiança
            label = f"acne: {confidence:.2f}"
            
            # Fundo para o texto
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), (0, 255, 0), -1)
            
            # Texto
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Adicionar informações gerais
        info_text = f"Acnes detectadas: {len(detections)}"
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # Salvar imagem
        cv2.imwrite(str(save_path), image)
        return True
    
    def extract_detection_crops(self, image_path: str, detections: list, base_name: str):
        """
        Extrair crops das regiões detectadas
        
        Args:
            image_path: Caminho da imagem original
            detections: Lista de detecções
            base_name: Nome base para salvar crops
        """
        full_path = self.images_dir / image_path.lstrip('/')
        
        # Carregar imagem
        image = cv2.imread(str(full_path))
        if image is None:
            return
        
        # Extrair cada detecção
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Verificar limites
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extrair crop
            crop = image[y1:y2, x1:x2]
            
            if crop.size > 0:
                crop_name = f"{base_name}_acne_{i+1:02d}_conf{confidence:.2f}.jpg"
                crop_path = self.output_dir / "detection_crops" / crop_name
                cv2.imwrite(str(crop_path), crop)
    
    def process_all_images(self, conf_threshold: float = 0.1, max_images: int = None):
        """
        Processar todas as imagens
        
        Args:
            conf_threshold: Threshold de confiança
            max_images: Número máximo de imagens (None = todas)
        """
        logger.info(f"Processando imagens com threshold de confiança: {conf_threshold}")
        
        # Lista de imagens para processar
        images_to_process = self.image_paths
        if max_images:
            images_to_process = images_to_process[:max_images]
        
        # Resultados
        all_results = []
        images_with_detections = 0
        total_detections = 0
        
        # Processar cada imagem
        for i, img_path in enumerate(tqdm(images_to_process, desc="Processando imagens")):
            try:
                # Detectar acnes
                result = self.detect_acnes_in_image(img_path, conf_threshold)
                
                if result is None:
                    continue
                
                all_results.append(result)
                
                # Contar estatísticas
                if result['has_acne']:
                    images_with_detections += 1
                    total_detections += result['num_detections']
                
                # Nome base para arquivos
                base_name = f"image_{i+1:06d}"
                
                # Salvar imagem com detecções ou sem detecções
                if result['has_acne']:
                    # Imagem COM detecções
                    save_path = self.output_dir / "images_with_detections" / f"{base_name}_detected.jpg"
                    self.draw_detections_on_image(img_path, result['detections'], save_path)
                    
                    # Extrair crops das detecções
                    self.extract_detection_crops(img_path, result['detections'], base_name)
                    
                else:
                    # Imagem SEM detecções - apenas copiar
                    full_path = self.images_dir / img_path.lstrip('/')
                    save_path = self.output_dir / "images_without_detections" / f"{base_name}_clean.jpg"
                    
                    if full_path.exists():
                        import shutil
                        shutil.copy(full_path, save_path)
                
            except Exception as e:
                logger.error(f"Erro ao processar {img_path}: {e}")
                continue
        
        # Salvar resultados em JSON
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'conf_threshold': conf_threshold,
            'total_images_processed': len(all_results),
            'images_with_detections': images_with_detections,
            'images_without_detections': len(all_results) - images_with_detections,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / len(all_results) if all_results else 0,
            'detection_rate': images_with_detections / len(all_results) if all_results else 0,
            'results': all_results
        }
        
        with open(self.output_dir / "detection_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Log das estatísticas
        logger.info("=== RESULTADOS DA DETECÇÃO ===")
        logger.info(f"Total de imagens processadas: {len(all_results)}")
        logger.info(f"Imagens com acnes: {images_with_detections}")
        logger.info(f"Imagens sem acnes: {len(all_results) - images_with_detections}")
        logger.info(f"Total de detecções: {total_detections}")
        logger.info(f"Taxa de detecção: {images_with_detections/len(all_results)*100:.1f}%")
        logger.info(f"Média de acnes por imagem: {total_detections/len(all_results):.2f}")
        
        return results_data
    
    def create_summary_report(self, results_data: dict):
        """
        Criar relatório resumido com visualizações
        
        Args:
            results_data: Dados dos resultados
        """
        logger.info("Criando relatório resumido...")
        
        # Dados para visualização
        total_images = results_data['total_images_processed']
        with_acne = results_data['images_with_detections']
        without_acne = results_data['images_without_detections']
        total_detections = results_data['total_detections']
        
        # Distribuição de detecções por imagem
        detections_per_image = [r['num_detections'] for r in results_data['results']]
        
        # Distribuição de confiança
        all_confidences = []
        for result in results_data['results']:
            for detection in result['detections']:
                all_confidences.append(detection['confidence'])
        
        # Criar visualizações
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Pizza - Imagens com/sem acne
        axes[0, 0].pie([with_acne, without_acne], 
                       labels=[f'Com acne ({with_acne})', f'Sem acne ({without_acne})'],
                       autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        axes[0, 0].set_title('Distribuição de Imagens')
        
        # Gráfico 2: Histograma - Detecções por imagem
        axes[0, 1].hist(detections_per_image, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Detecções por Imagem')
        axes[0, 1].set_xlabel('Número de Detecções')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Histograma - Confiança das detecções
        if all_confidences:
            axes[1, 0].hist(all_confidences, bins=20, edgecolor='black', alpha=0.7, color='green')
            axes[1, 0].set_title('Distribuição de Confiança')
            axes[1, 0].set_xlabel('Confiança')
            axes[1, 0].set_ylabel('Frequência')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Nenhuma detecção', ha='center', va='center')
            axes[1, 0].set_title('Distribuição de Confiança')
        
        # Gráfico 4: Estatísticas gerais
        stats = [
            f'Total de imagens: {total_images}',
            f'Com acnes: {with_acne} ({with_acne/total_images*100:.1f}%)',
            f'Sem acnes: {without_acne} ({without_acne/total_images*100:.1f}%)',
            f'Total de detecções: {total_detections}',
            f'Média por imagem: {total_detections/total_images:.2f}',
            f'Confiança média: {np.mean(all_confidences):.3f}' if all_confidences else 'N/A'
        ]
        
        axes[1, 1].axis('off')
        for i, stat in enumerate(stats):
            axes[1, 1].text(0.1, 0.9-i*0.12, stat, fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Estatísticas Gerais')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "detection_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Relatório HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Detecção de Acnes</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .stat {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Relatório de Detecção de Acnes</h1>
                <p><strong>Data:</strong> {results_data['timestamp']}</p>
                <p><strong>Modelo usado:</strong> {results_data['model_path']}</p>
                <p><strong>Threshold de confiança:</strong> {results_data['conf_threshold']}</p>
            </div>
            
            <div class="section">
                <h2>📊 Estatísticas Gerais</h2>
                <div class="stat"><strong>Total de imagens processadas:</strong> {total_images}</div>
                <div class="stat"><strong>Imagens com acnes detectadas:</strong> <span class="success">{with_acne}</span> ({with_acne/total_images*100:.1f}%)</div>
                <div class="stat"><strong>Imagens sem acnes:</strong> {without_acne} ({without_acne/total_images*100:.1f}%)</div>
                <div class="stat"><strong>Total de detecções:</strong> <span class="success">{total_detections}</span></div>
                <div class="stat"><strong>Média de acnes por imagem:</strong> {total_detections/total_images:.2f}</div>
                <div class="stat"><strong>Taxa de detecção:</strong> {with_acne/total_images*100:.1f}%</div>
        """
        
        if all_confidences:
            html_content += f"""
                <div class="stat"><strong>Confiança média:</strong> {np.mean(all_confidences):.3f}</div>
                <div class="stat"><strong>Confiança mínima:</strong> {np.min(all_confidences):.3f}</div>
                <div class="stat"><strong>Confiança máxima:</strong> {np.max(all_confidences):.3f}</div>
            """
        
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>📁 Arquivos Gerados</h2>
                <ul>
                    <li><strong>Imagens com detecções:</strong> {self.output_dir}/images_with_detections/ ({with_acne} imagens)</li>
                    <li><strong>Imagens sem detecções:</strong> {self.output_dir}/images_without_detections/ ({without_acne} imagens)</li>
                    <li><strong>Crops das detecções:</strong> {self.output_dir}/detection_crops/ ({total_detections} crops)</li>
                    <li><strong>Resultados JSON:</strong> {self.output_dir}/detection_results.json</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📈 Visualizações</h2>
                <img src="detection_summary.png" alt="Gráficos de Detecção">
            </div>
            
        </body>
        </html>
        """
        
        with open(self.output_dir / "detection_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Relatório salvo em: {self.output_dir / 'detection_report.html'}")


def main():
    """Função principal"""
    
    # Configurações
    MODEL_PATH = "trained_model/acne_model/weights/best.pt"
    IMAGES_DIR = "images"
    CSV_PATH = "df.csv"
    OUTPUT_DIR = "detection_results"
    
    # Verificações
    if not Path(MODEL_PATH).exists():
        logger.error(f"Modelo não encontrado: {MODEL_PATH}")
        logger.error("Execute primeiro train_manual_model.py")
        return
    
    print("=== APLICAÇÃO DO MODELO PARA DETECÇÃO DE ACNES ===")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Imagens: {IMAGES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Configurações de detecção
    conf_threshold = 0.1  # Threshold de confiança
    max_images = None  # Processar todas as imagens (ou definir um número)
    
    print(f"Confiança mínima: {conf_threshold}")
    print(f"Máximo de imagens: {max_images or 'Todas'}")
    print()
    
    # Inicializar aplicador
    applicator = AcneModelApplicator(MODEL_PATH, IMAGES_DIR, CSV_PATH, OUTPUT_DIR)
    
    input("Pressione ENTER para iniciar o processamento...")
    
    # Processar todas as imagens
    results = applicator.process_all_images(
        conf_threshold=conf_threshold,
        max_images=max_images
    )
    
    # Criar relatório
    applicator.create_summary_report(results)
    
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print(f"- Imagens com detecções: {OUTPUT_DIR}/images_with_detections/")
    print(f"- Imagens sem detecções: {OUTPUT_DIR}/images_without_detections/")
    print(f"- Crops das acnes: {OUTPUT_DIR}/detection_crops/")
    print(f"- Relatório: {OUTPUT_DIR}/detection_report.html")


if __name__ == "__main__":
    main() 
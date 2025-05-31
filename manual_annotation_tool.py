import cv2
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManualAnnotationTool:
    def __init__(self, images_dir: str, csv_path: str, output_dir: str = "manual_annotations"):
        """
        Ferramenta para anotação manual de espinhas
        
        Args:
            images_dir: Diretório das imagens
            csv_path: Arquivo CSV com lista de imagens
            output_dir: Diretório para salvar anotações
        """
        self.images_dir = Path(images_dir)
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Carregar lista de imagens
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df['files'].tolist()
        
        # Estado da anotação
        self.current_image = None
        self.current_bboxes = []
        self.drawing = False
        self.start_point = None
        self.temp_bbox = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos do mouse"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Começar a desenhar bounding box
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Atualizar bounding box temporário
                self.temp_bbox = (self.start_point[0], self.start_point[1], x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finalizar bounding box
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Calcular coordenadas do bounding box
                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])
                
                # Verificar se o bounding box tem tamanho mínimo
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    # Converter para coordenadas normalizadas (formato YOLO)
                    h, w = self.current_image.shape[:2]
                    
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Adicionar bounding box
                    bbox = {
                        'class': 0,  # classe 'acne'
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'pixel_coords': [x1, y1, x2, y2]
                    }
                    
                    self.current_bboxes.append(bbox)
                    logger.info(f"Bounding box adicionado: {len(self.current_bboxes)} total")
                
                self.temp_bbox = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remover último bounding box
            if self.current_bboxes:
                removed = self.current_bboxes.pop()
                logger.info(f"Bounding box removido. Restam: {len(self.current_bboxes)}")
    
    def draw_bboxes(self, image, current_image=0, total_images=1):
        """Desenhar bounding boxes na imagem"""
        display_image = image.copy()
        h, w = image.shape[:2]
        
        # Desenhar bounding boxes salvos
        for bbox in self.current_bboxes:
            x1, y1, x2, y2 = bbox['pixel_coords']
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, 'acne', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Desenhar bounding box temporário
        if self.temp_bbox:
            x1, y1, x2, y2 = self.temp_bbox
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Adicionar instruções
        instructions = [
            "INSTRUCOES:",
            "- Clique e arraste para marcar ESPINHA",
            "- Botao direito: remove ultimo box",  
            "- ENTER: salva e proxima imagem",
            "- ESC: sair e salvar progresso",
            "- S: pular esta imagem",
            f"- Imagem {current_image + 1}/{total_images}",
            f"- Marcadas: {len(self.current_bboxes)} espinhas"
        ]
        
        y_offset = 30
        for instruction in instructions:
            # Fundo para o texto
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_image, (10, y_offset - text_size[1] - 5), 
                         (15 + text_size[0], y_offset + 5), (0, 0, 0), -1)
            
            # Texto
            cv2.putText(display_image, instruction, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
        return display_image
    
    def annotate_images(self, start_index: int = 0, max_images: int = None):
        """
        Iniciar processo de anotação manual
        
        Args:
            start_index: Índice para começar
            max_images: Número máximo de imagens para anotar
        """
        logger.info("=== FERRAMENTA DE ANOTAÇÃO MANUAL ===")
        logger.info("Use o mouse para marcar regiões com espinhas")
        
        # Criar arquivo para salvar anotações
        annotations_file = self.output_dir / "annotations.json"
        all_annotations = []
        
        # Carregar anotações existentes se houver
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                all_annotations = json.load(f)
            logger.info(f"Carregadas {len(all_annotations)} anotações existentes")
        
        # Lista de imagens para anotar
        images_to_annotate = self.image_paths[start_index:]
        if max_images:
            images_to_annotate = images_to_annotate[:max_images]
        
        logger.info(f"Anotando {len(images_to_annotate)} imagens")
        
        # Criar janela única
        window_name = 'Anotacao de Espinhas'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        current_idx = 0
        
        while current_idx < len(images_to_annotate):
            img_path = images_to_annotate[current_idx]
            full_path = self.images_dir / img_path.lstrip('/')
            
            # Verificar se imagem já foi anotada
            existing_annotation = None
            for ann in all_annotations:
                if ann['image_path'] == img_path:
                    existing_annotation = ann
                    break
            
            logger.info(f"Imagem {current_idx + 1}/{len(images_to_annotate)}: {img_path}")
            
            # Carregar imagem
            if not full_path.exists():
                logger.error(f"Imagem não encontrada: {full_path}")
                current_idx += 1
                continue
            
            image = cv2.imread(str(full_path))
            if image is None:
                logger.error(f"Erro ao carregar imagem: {full_path}")
                current_idx += 1
                continue
            
            # Pipeline de pré-processamento explícito
            original_shape = image.shape
            logger.info(f"Imagem original: {original_shape[1]}x{original_shape[0]}")
            
            # Verificação de qualidade da imagem
            if len(image.shape) != 3:
                logger.warning(f"Imagem não é RGB: {full_path}")
                continue
            
            # Redimensionamento inteligente mantendo aspect ratio
            h, w = image.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"Imagem redimensionada: {new_w}x{new_h} (fator: {scale:.3f})")
            
            # Normalização de contraste (opcional, para melhor visualização)
            # Converte para LAB, equaliza canal L, converte de volta
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_channels = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab_channels[0] = clahe.apply(lab_channels[0])
            lab = cv2.merge(lab_channels)
            image_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Usar imagem normalizada apenas para visualização, manter original para anotação
            display_image = image_normalized
            annotation_image = image  # Imagem original para coordenadas precisas
            
            self.current_image = annotation_image
            self.current_bboxes = []
            
            # Carregar bounding boxes existentes se houver
            if existing_annotation:
                logger.info(f"Imagem já anotada com {len(existing_annotation['bboxes'])} bounding boxes")
                self.current_bboxes = existing_annotation['bboxes'].copy()
                # Recalcular coordenadas de pixel
                h, w = annotation_image.shape[:2]
                for bbox in self.current_bboxes:
                    x_center = bbox['x_center'] * w
                    y_center = bbox['y_center'] * h
                    width = bbox['width'] * w
                    height = bbox['height'] * h
                    
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    bbox['pixel_coords'] = [x1, y1, x2, y2]
            
            # Loop de anotação para esta imagem
            while True:
                display_image = self.draw_bboxes(display_image, current_image=current_idx, total_images=len(images_to_annotate))
                cv2.imshow(window_name, display_image)
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == 13:  # ENTER - salvar e próxima
                    # Salvar anotação
                    annotation_data = {
                        'image_path': img_path,
                        'image_id': start_index + current_idx,
                        'bboxes': [{
                            'class': bbox['class'],
                            'x_center': bbox['x_center'],
                            'y_center': bbox['y_center'],
                            'width': bbox['width'],
                            'height': bbox['height']
                        } for bbox in self.current_bboxes],
                        'num_acnes': len(self.current_bboxes)
                    }
                    
                    # Remover anotação existente se houver
                    all_annotations = [ann for ann in all_annotations if ann['image_path'] != img_path]
                    all_annotations.append(annotation_data)
                    
                    logger.info(f"Anotação salva: {len(self.current_bboxes)} bounding boxes")
                    break
                
                elif key == 27:  # ESC - sair
                    logger.info("Saindo da anotação...")
                    cv2.destroyAllWindows()
                    
                    # Salvar anotações antes de sair
                    with open(annotations_file, 'w') as f:
                        json.dump(all_annotations, f, indent=2)
                    
                    return all_annotations
                
                elif key == ord('s'):  # S - pular imagem
                    logger.info("Pulando imagem...")
                    break
            
            current_idx += 1
            
            # Salvar progresso a cada 5 imagens
            if current_idx % 5 == 0:
                with open(annotations_file, 'w') as f:
                    json.dump(all_annotations, f, indent=2)
                logger.info(f"Progresso salvo: {len(all_annotations)} anotações")
        
        cv2.destroyAllWindows()
        
        # Salvar anotações finais
        with open(annotations_file, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        logger.info(f"Anotação concluída! Total: {len(all_annotations)} imagens anotadas")
        return all_annotations
    
    def create_yolo_dataset(self, annotations_data: list, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Criar dataset no formato YOLO a partir das anotações manuais
        
        Args:
            annotations_data: Lista de anotações
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação
        """
        logger.info("Criando dataset YOLO...")
        
        # Criar diretórios
        dataset_dir = self.output_dir / "yolo_dataset"
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Filtrar apenas imagens com anotações
        annotated_images = [ann for ann in annotations_data if ann['num_acnes'] > 0]
        
        if len(annotated_images) == 0:
            logger.error("Nenhuma imagem com anotações encontrada!")
            return None
        
        # Dividir dataset
        np.random.seed(42)
        np.random.shuffle(annotated_images)
        
        n_total = len(annotated_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = annotated_images[:n_train]
        val_data = annotated_images[n_train:n_train + n_val]
        test_data = annotated_images[n_train + n_val:]
        
        # Processar cada split
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            logger.info(f"Processando {split_name}: {len(split_data)} imagens")
            
            for i, ann_data in enumerate(split_data):
                # Copiar imagem
                img_path = ann_data['image_path']
                full_path = self.images_dir / img_path.lstrip('/')
                
                if not full_path.exists():
                    continue
                
                # Nome do arquivo
                img_name = f"{split_name}_{i:06d}.jpg"
                label_name = f"{split_name}_{i:06d}.txt"
                
                # Copiar imagem
                import shutil
                shutil.copy(full_path, dataset_dir / split_name / "images" / img_name)
                
                # Criar arquivo de label
                with open(dataset_dir / split_name / "labels" / label_name, 'w') as f:
                    for bbox in ann_data['bboxes']:
                        line = f"{bbox['class']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                        f.write(line)
        
        # Criar arquivo de configuração YOLO
        config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': 1,
            'names': ['acne']
        }
        
        config_path = dataset_dir / "acne_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Salvar informações da divisão
        split_info = {
            'total_images': len(annotated_images),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1 - train_ratio - val_ratio
        }
        
        with open(dataset_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Dataset YOLO criado:")
        logger.info(f"- Treino: {len(train_data)} imagens")
        logger.info(f"- Validação: {len(val_data)} imagens") 
        logger.info(f"- Teste: {len(test_data)} imagens")
        logger.info(f"- Configuração: {config_path}")
        
        return config_path


def main():
    """Função principal para anotação manual"""
    
    IMAGES_DIR = "images"
    CSV_PATH = "df.csv"
    OUTPUT_DIR = "manual_annotations"
    
    # Inicializar ferramenta
    tool = ManualAnnotationTool(IMAGES_DIR, CSV_PATH, OUTPUT_DIR)
    
    # Parâmetros
    start_index = 0
    max_images = 50  # Anotar 50 imagens inicialmente
    
    print("=== FERRAMENTA DE ANOTAÇÃO MANUAL DE ESPINHAS ===")
    print(f"Imagens encontradas: {len(tool.image_paths)}")
    print(f"Vamos anotar {max_images} imagens")
    print()
    print("INSTRUÇÕES:")
    print("- Clique e arraste para marcar uma espinha")
    print("- Botão direito do mouse: remove último bounding box")
    print("- ENTER: salva e vai para próxima imagem")
    print("- ESC: sair e salvar progresso")
    print("- S: pular imagem atual")
    print()
    
    input("Pressione ENTER para começar...")
    
    # Fazer anotações
    annotations = tool.annotate_images(start_index=start_index, max_images=max_images)
    
    # Criar dataset YOLO
    if annotations:
        config_path = tool.create_yolo_dataset(annotations)
        print(f"\nDataset YOLO criado: {config_path}")
        print("Próximo passo: executar train_manual_model.py")


if __name__ == "__main__":
    main() 
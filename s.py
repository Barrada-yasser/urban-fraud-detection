from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# Configuration
class Config:
    # Chemins des modèles
    MODEL_STANDARD = "yolov8s-seg.pt"  # Pour véhicules et personnes
    MODEL_DETECT = "yolov8s.pt"        # Pour boîtes englobantes
    MODEL_ROAD = "./runs/train/exp/road_seg2/weights/best.pt"  # Pour routes
    MODEL_CROSSWALK = "./runs/train/exp/passage_pieton_seg3/weights/best.pt"  # Pour passages piétons
    
    # Dossiers
    TEST_FOLDER = "./JAAD_frames/video_0019"
    RESULTS_FOLDER = "./combined_results"
    
    # Classes d'intérêt dans YOLO
    CLASSES = {
        0: "personne",
        2: "voiture",
        3: "moto",
        5: "bus",
        7: "camion",
        9: "feu_trafic"
    }
    
    # Couleurs pour chaque type d'objet
    COLORS = {
        "personne": (0, 0, 255),       # Rouge
        "voiture": (255, 0, 0),        # Bleu
        "moto": (255, 0, 0),           # Bleu
        "bus": (255, 0, 0),            # Bleu
        "camion": (255, 0, 0),         # Bleu
        "feu_trafic": (255, 165, 0),   # Orange
        "route": (0, 255, 0),          # Vert
        "passage_pieton": (165, 42, 42)  # Marron
    }
    
    # Seuils pour détection des couleurs
    COLOR_THRESHOLDS = {
        "rouge": [(0, 0, 200), (50, 50, 255)],      # Piétons
        "bleu": [(200, 0, 0), (255, 50, 50)],       # Véhicules
        "vert": [(0, 200, 0), (50, 255, 50)],       # Routes
        "marron": [(20, 20, 140), (70, 70, 190)]    # Passages piétons
    }
    
    # Extensions d'images valides
    VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Seuil minimal de pixels pour la détection
    MIN_PIXELS = 200
    
    # Seuils pour la classification finale
    SITUATION_THRESHOLDS = {
        "✅ Piéton traverse correctement sur le passage piéton": 4,
        "❌ DANGER: Piéton traverse la route hors du passage piéton": 6,
        "🟡 Route vide avec passage piéton visible": 3,
        "🚗 Route occupée par des véhicules uniquement": 3,
        "🟢 Route totalement vide": 3
    }

# Fonctions utilitaires
def verify_models():
    """Vérifie l'existence des modèles et utilise des alternatives si nécessaire"""
    if not os.path.exists(Config.MODEL_ROAD):
        Config.MODEL_ROAD = Config.MODEL_STANDARD
        print("Modèle de route non trouvé, utilisation du modèle standard")
    else:
        print(f"Utilisation du modèle de route: {Config.MODEL_ROAD}")
    
    if not os.path.exists(Config.MODEL_CROSSWALK):
        Config.MODEL_CROSSWALK = Config.MODEL_STANDARD
        print("Modèle de passage piéton non trouvé, utilisation du modèle standard")
    else:
        print(f"Utilisation du modèle de passage piéton: {Config.MODEL_CROSSWALK}")

def apply_mask(image, mask, color, alpha=0.5):
    """Applique un masque avec une couleur spécifique"""
    mask_img = mask.cpu().numpy().astype(np.uint8) * 255
    mask_img = cv2.resize(mask_img, (image.shape[1], image.shape[0]))
    
    for c in range(3):
        image[:, :, c] = np.where(mask_img > 0, 
                              image[:, :, c] * (1 - alpha) + color[c] * alpha,
                              image[:, :, c])
    return image

def analyze_situation(masks):
    """Analyse la situation basée sur les intersections de masques"""
    # Calculer les intersections
    pieton_route = cv2.bitwise_and(masks["rouge"], masks["vert"])
    pieton_passage = cv2.bitwise_and(masks["rouge"], masks["marron"]) 
    pieton_passage_route = cv2.bitwise_and(pieton_passage, masks["vert"])
    passage_route = cv2.bitwise_and(masks["marron"], masks["vert"])
    vehicule_route = cv2.bitwise_and(masks["bleu"], masks["vert"])
    
    # Vérifier la présence de chaque élément
    has_pieton = cv2.countNonZero(masks["rouge"]) > Config.MIN_PIXELS
    has_vehicule = cv2.countNonZero(masks["bleu"]) > Config.MIN_PIXELS
    has_route = cv2.countNonZero(masks["vert"]) > Config.MIN_PIXELS
    has_passage = cv2.countNonZero(masks["marron"]) > Config.MIN_PIXELS
    
    # Vérifier les intersections
    has_pieton_route = cv2.countNonZero(pieton_route) > Config.MIN_PIXELS
    has_pieton_passage = cv2.countNonZero(pieton_passage) > Config.MIN_PIXELS
    has_pieton_passage_route = cv2.countNonZero(pieton_passage_route) > Config.MIN_PIXELS
    has_passage_route = cv2.countNonZero(passage_route) > Config.MIN_PIXELS
    has_vehicule_route = cv2.countNonZero(vehicule_route) > Config.MIN_PIXELS
    
    # Déterminer la situation
    situation = "Indéterminée"
    situation_color = (200, 200, 200)  # Gris par défaut
    
    if has_pieton_passage_route:
        situation = "✅ Piéton traverse correctement sur le passage piéton"
        situation_color = (0, 255, 0)  # Vert
    elif has_pieton_route and not has_pieton_passage:
        situation = "❌ DANGER: Piéton traverse la route hors du passage piéton"
        situation_color = (0, 0, 255)  # Rouge
    elif has_passage_route and not has_pieton and not has_vehicule:
        situation = "🟡 Route vide avec passage piéton visible"
        situation_color = (0, 255, 255)  # Jaune
    elif has_vehicule_route and not has_pieton and not has_passage:
        situation = "🚗 Route occupée par des véhicules uniquement"
        situation_color = (255, 0, 0)  # Bleu
    elif has_route and not has_pieton and not has_vehicule and not has_passage:
        situation = "🟢 Route totalement vide"
        situation_color = (0, 128, 0)  # Vert foncé
    
    return situation, situation_color

def determine_final_situation(situation_counts):
    """Détermine la situation finale basée sur les comptages des situations"""
    for situation, threshold in Config.SITUATION_THRESHOLDS.items():
        if situation_counts.get(situation, 0) >= threshold:
            return situation
    
    # Si aucun seuil n'est atteint, retourner la situation la plus fréquente
    if situation_counts:
        return situation_counts.most_common(1)[0][0]
    else:
        return "Indéterminée"

class RoadAnalyzer:
    def __init__(self):
        """Initialise l'analyseur de routes avec les modèles YOLO"""
        verify_models()
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
        
        # Charger les modèles
        print("Chargement des modèles...")
        self.model_standard = YOLO(Config.MODEL_STANDARD)
        self.model_detect = YOLO(Config.MODEL_DETECT)
        self.model_road = YOLO(Config.MODEL_ROAD)
        self.model_crosswalk = YOLO(Config.MODEL_CROSSWALK)
        print("Modèles chargés avec succès")
        
        # Compteur de situations
        self.situation_counts = Counter()

    def process_images(self, max_images=None):
        """Traite les images du dossier de test"""
        # Trouver toutes les images valides
        image_files = [f for f in os.listdir(Config.TEST_FOLDER) 
                      if any(f.lower().endswith(ext) for ext in Config.VALID_EXTENSIONS)]
        
        if not image_files:
            print(f"Aucune image trouvée dans {Config.TEST_FOLDER}")
            return
        
        # Trier les images par nom (important pour les séquences)
        image_files.sort()
        
        # Limiter le nombre d'images à traiter si spécifié
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Traitement de {len(image_files)} images...")
        
        # Réinitialiser le compteur de situations
        self.situation_counts = Counter()
        
        # Traiter chaque image
        for image_file in image_files:
            situation = self.process_single_image(image_file)
            if situation != "Indéterminée":
                self.situation_counts[situation] += 1
        
        # Déterminer la situation finale
        final_situation = determine_final_situation(self.situation_counts)
        
        # Créer une image récapitulative avec la classe finale
        self.create_summary_image(image_files, final_situation)
        
        print(f"Traitement terminé! Les résultats sont dans: {Config.RESULTS_FOLDER}")
        print(f"Comptage des situations: {dict(self.situation_counts)}")
        print(f"Classification finale: {final_situation}")
    
    def process_single_image(self, image_file):
        """Traite une seule image avec tous les modèles et analyse la situation"""
        # Chemin complet de l'image
        image_path = os.path.join(Config.TEST_FOLDER, image_file)
        print(f"Traitement de: {image_path}")
        
        # Charger l'image originale
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Erreur lors de la lecture de l'image: {image_path}")
            return "Indéterminée"
        
        # Créer une copie pour la visualisation
        combined_image = original_image.copy()
        
        # Initialiser
        legend = []
        analysis_masks = {
            "rouge": np.zeros_like(original_image[:, :, 0]),  # Piétons
            "bleu": np.zeros_like(original_image[:, :, 0]),   # Véhicules
            "vert": np.zeros_like(original_image[:, :, 0]),   # Routes
            "marron": np.zeros_like(original_image[:, :, 0])  # Passages piétons
        }
        
        # 1. Appliquer le modèle standard (personnes et véhicules)
        self.apply_standard_model(image_path, combined_image, analysis_masks, legend)
        
        # 2. Appliquer le modèle de route
        self.apply_road_model(image_path, combined_image, analysis_masks, legend)
        
        # 3. Appliquer le modèle de passage piéton
        self.apply_crosswalk_model(image_path, combined_image, analysis_masks, legend)
        
        # 4. Ajouter les boîtes englobantes avec le modèle de détection
        self.apply_detection_model(image_path, combined_image, legend)
        
        # 5. Analyser la situation
        situation, situation_color = analyze_situation(analysis_masks)
        
        # Ajouter la situation en haut de l'image
        cv2.putText(combined_image, situation, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, situation_color, 2)
        
        # Ajouter les légendes
        self.add_legend(combined_image, legend)
        
        # Sauvegarder l'image avec tous les masques combinés
        output_path = os.path.join(Config.RESULTS_FOLDER, f"combined_{image_file}")
        cv2.imwrite(output_path, combined_image)
        
        # Sauvegarder une image avec juste l'analyse de situation
        analysis_img = original_image.copy()
        cv2.putText(analysis_img, situation, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, situation_color, 2)
        analysis_path = os.path.join(Config.RESULTS_FOLDER, f"analysis_{image_file}")
        cv2.imwrite(analysis_path, analysis_img)
        
        print(f"Résultat sauvegardé: {output_path}")
        print(f"Situation détectée: {situation}")
        
        return situation
    
    def create_summary_image(self, image_files, final_situation):
        """Crée une image récapitulative avec la situation finale en utilisant PIL pour gérer les émojis"""
        if not image_files:
            return
            
        # Créer une image avec PIL
        width, height = 800, 400
        # Créer une image noire (fond noir)
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Essayer de charger une police qui supporte les émojis (si disponible)
        try:
            # Sur Windows, essayez Segoe UI Emoji qui supporte les émojis
            font_title = ImageFont.truetype("seguiemj.ttf", 36)
            font_regular = ImageFont.truetype("seguiemj.ttf", 24)
            font_small = ImageFont.truetype("seguiemj.ttf", 20)
            font_final = ImageFont.truetype("seguiemj.ttf", 28)
        except IOError:
            # Fallback sur une police par défaut
            font_title = ImageFont.load_default()
            font_regular = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_final = ImageFont.load_default()
        
        # Définir la couleur en fonction de la situation
        if "✅" in final_situation:
            color = (0, 255, 0)  # Vert
        elif "❌" in final_situation:
            color = (255, 0, 0)  # Rouge
        elif "🟡" in final_situation:
            color = (255, 255, 0)  # Jaune
        elif "🚗" in final_situation:
            color = (0, 0, 255)  # Bleu
        elif "🟢" in final_situation:
            color = (0, 128, 0)  # Vert foncé
        else:
            color = (200, 200, 200)  # Gris
        
        # Ajouter le titre
        draw.text((20, 20), "Analyse de séquence", fill=(255, 255, 255), font=font_title)
        
        # Ajouter le nombre d'images traitées
        draw.text((20, 80), f"Nombre d'images traitées: {len(image_files)}", fill=(255, 255, 255), font=font_regular)
        
        # Ajouter les comptages de situations
        y_pos = 130
        for situation, count in self.situation_counts.most_common():
            draw.text((20, y_pos), f"{situation}: {count} images", fill=(255, 255, 255), font=font_small)
            y_pos += 30
        
        # Ajouter la situation finale
        draw.text((20, 280), "CLASSIFICATION FINALE:", fill=(255, 255, 255), font=font_regular)
        draw.text((20, 320), final_situation, fill=color, font=font_final)
        
        # Sauvegarder l'image récapitulative
        summary_path = os.path.join(Config.RESULTS_FOLDER, "classification_finale.jpg")
        image.save(summary_path)
        print(f"Résumé de classification sauvegardé: {summary_path}")
    
    def apply_standard_model(self, image_path, combined_image, analysis_masks, legend):
        """Applique le modèle standard pour détecter les personnes et véhicules"""
        results = self.model_standard(image_path, task="segment")
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks) > 0:
                for i, (mask, cls) in enumerate(zip(r.masks.data, r.boxes.cls)):
                    cls_id = int(cls.item())
                    if cls_id in Config.CLASSES:
                        class_name = Config.CLASSES[cls_id]
                        color = Config.COLORS[class_name]
                        combined_image = apply_mask(combined_image, mask, color)
                        
                        # Ajouter au masque d'analyse
                        mask_img = mask.cpu().numpy().astype(np.uint8) * 255
                        mask_img = cv2.resize(mask_img, (combined_image.shape[1], combined_image.shape[0]))
                        
                        if class_name == "personne":
                            analysis_masks["rouge"] = cv2.bitwise_or(analysis_masks["rouge"], mask_img)
                        elif class_name in ["voiture", "moto", "bus", "camion"]:
                            analysis_masks["bleu"] = cv2.bitwise_or(analysis_masks["bleu"], mask_img)
                        
                        if class_name not in legend:
                            legend.append(class_name)
    
    def apply_road_model(self, image_path, combined_image, analysis_masks, legend):
        """Applique le modèle de détection de route"""
        results = self.model_road(image_path, task="segment")
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks) > 0:
                for mask in r.masks.data:
                    combined_image = apply_mask(combined_image, mask, Config.COLORS["route"], alpha=0.3)
                    
                    # Ajouter au masque d'analyse
                    mask_img = mask.cpu().numpy().astype(np.uint8) * 255
                    mask_img = cv2.resize(mask_img, (combined_image.shape[1], combined_image.shape[0]))
                    analysis_masks["vert"] = cv2.bitwise_or(analysis_masks["vert"], mask_img)
                
                if "route" not in legend:
                    legend.append("route")
    
    def apply_crosswalk_model(self, image_path, combined_image, analysis_masks, legend):
        """Applique le modèle de détection de passage piéton"""
        results = self.model_crosswalk(image_path, task="segment")
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks) > 0:
                for mask in r.masks.data:
                    combined_image = apply_mask(combined_image, mask, Config.COLORS["passage_pieton"], alpha=0.4)
                    
                    # Ajouter au masque d'analyse
                    mask_img = mask.cpu().numpy().astype(np.uint8) * 255
                    mask_img = cv2.resize(mask_img, (combined_image.shape[1], combined_image.shape[0]))
                    analysis_masks["marron"] = cv2.bitwise_or(analysis_masks["marron"], mask_img)
                
                if "passage_pieton" not in legend:
                    legend.append("passage_pieton")
    
    def apply_detection_model(self, image_path, combined_image, legend):
        """Applique le modèle de détection pour ajouter les boîtes englobantes"""
        results = self.model_detect(image_path, task="detect")
        for r in results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    cls_id = int(cls.item())
                    confidence = float(conf.item())
                    
                    if cls_id in Config.CLASSES:
                        class_name = Config.CLASSES[cls_id]
                        color = Config.COLORS[class_name]
                        
                        # Extraire les coordonnées de la boîte
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        
                        # Dessiner la boîte englobante
                        cv2.rectangle(combined_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Ajouter le texte avec la classe et la confiance
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(combined_image, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if f"{class_name} (boîte)" not in legend and class_name not in legend:
                            legend.append(f"{class_name} (boîte)")
    
    def add_legend(self, image, legend):
        """Ajoute la légende à l'image"""
        y_pos = 60
        for item in legend:
            if "(boîte)" in item:
                color = Config.COLORS[item.split(" ")[0]]
            else:
                color = Config.COLORS[item]
            cv2.putText(image, item, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 25

# Point d'entrée principal
if __name__ == "__main__":
    analyzer = RoadAnalyzer()
    analyzer.process_images() 
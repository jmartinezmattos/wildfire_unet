import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import warnings

class FireDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, max_images=None):
        print("Image dir: ", image_dir)
        """
        Dataset para segmentación de incendios.
        - image_dir: carpeta con las imágenes
        - mask_dir: carpeta con las máscaras
        - transform: transformaciones (Albumentations o similares)
        - max_images: número máximo de imágenes a cargar (opcional)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        print("Total imágenes encontradas: ", len(self.images))
        self.valid_images = []
        for img_name in self.images:
            mask_name = img_name.replace(".png", "_gt.png")
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.valid_images.append(img_name)
            else:
                warnings.warn(f"No se encontró la máscara para {img_name}, se omitirá.")

        # Limitar cantidad de imágenes cargadas
        if max_images is not None:
            if max_images < len(self.valid_images):
                self.valid_images = self.valid_images[:max_images]
                print(f"🔹 Cargando solo las primeras {max_images} imágenes válidas.")
            else:
                print(f"🔹 Se encontraron {len(self.valid_images)} imágenes válidas (todas se cargarán).")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, index):
        img_name = self.valid_images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".png", "_gt.png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

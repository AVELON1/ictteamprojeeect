import cv2
import numpy as np

def apply_rotation(img, angle):
    """Rotates image by specified angle around center without cropping."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_blur(img, ksize):
    """Applies Gaussian blur. ksize must be odd integer >= 1."""
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_brightness_contrast(img, alpha=1.0, beta=0):
    """
    Adjusts contrast (alpha: 0.5 to 1.5) and brightness (beta: -50 to 50).
    new_img = alpha * img + beta
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def apply_noise(img, sigma=15):
    """Adds Gaussian sensor noise."""
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def generate_synthetic_variants(img, num_variants=3):
    """
    Generates realistic adverse-condition variations of a single road sign:
    - Slight rotation (±10° to ±15°)
    - Optical defocus / motion blur (GaussianBlur)
    - Dusk / glare (brightness & contrast changes)
    - Sensor grain (Gaussian noise)
    """
    variants = []
    for _ in range(num_variants):
        var = img.copy()
        
        # 1. Random rotation ±15 degrees
        angle = np.random.uniform(-15.0, 15.0)
        var = apply_rotation(var, angle)
        
        # 2. Random blur (50% chance)
        if np.random.rand() > 0.3:
            ksize = np.random.choice([3, 5, 7, 9])
            var = apply_blur(var, ksize)
            
        # 3. Random lighting / contrast (dusk/darkness vs glare)
        alpha = np.random.uniform(0.6, 1.3)
        beta = int(np.random.uniform(-35, 25))
        var = apply_brightness_contrast(var, alpha=alpha, beta=beta)
        
        # 4. Sensor noise
        if np.random.rand() > 0.4:
            sigma = np.random.uniform(5, 20)
            var = apply_noise(var, sigma)
            
        variants.append(var)
        
    return variants

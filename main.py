import os
import cv2
import numpy as np
import albumentations as A

# --- Custom Salt & Pepper Noise ---
def add_salt_pepper_noise(image, amount=0.1, salt_ratio=0.95):
    noisy = image.copy()
    row, col, ch = noisy.shape
    total = amount * row * col

    num_salt = int(total * salt_ratio)
    num_pepper = int(total * (1 - salt_ratio))

    coords = [np.random.randint(0, i - 1, num_salt) for i in (row, col)]
    noisy[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in (row, col)]
    noisy[coords[0], coords[1], :] = 0

    return noisy

# --- Adaptive Weather Effects ---
def get_adaptive_rain_params(image_shape):
    """Get rain parameters based on image size"""
    height, width = image_shape[:2]
    base_size = min(height, width)
    
    # Scale parameters based on image size (assuming 800x800 as base)
    scale_factor = base_size / 800.0
    
    return {
        'drop_length': max(10, int(15 * scale_factor)),
        'drop_width': max(1, int(2 * scale_factor)),
        'blur_value': max(2, int(3 * scale_factor))
    }

def get_adaptive_sunflare_params(image_shape):
    """Get sun flare parameters based on image size"""
    height, width = image_shape[:2]
    base_size = min(height, width)
    
    # Scale parameters based on image size (moderately increased effect)
    scale_factor = base_size / 800.0
    
    # Moderately increased values for all image sizes
    if base_size < 400:
        min_radius = max(5, int(base_size * 0.015))  # 1.5% of image size
    else:
        min_radius = 50  # Moderate minimum radius
        
    return {
        'src_radius': max(min_radius, int(0.75 * scale_factor))  # Moderately increased
    }

def apply_adaptive_rain(image, rain_type='drizzle', intensity='light'):
    """Apply rain effect with adaptive parameters"""
    params = get_adaptive_rain_params(image.shape)
    
    if intensity == 'light':
        brightness_coeff = 0.8
        params['drop_length'] = int(params['drop_length'] * 0.8)
    else:  # heavy
        brightness_coeff = 0.5
        params['drop_length'] = int(params['drop_length'] * 1.5)
        params['drop_width'] = int(params['drop_width'] * 1.5)
    
    rain_transform = A.RandomRain(
        drop_length=params['drop_length'],
        drop_width=params['drop_width'],
        blur_value=params['blur_value'],
        brightness_coefficient=brightness_coeff,
        rain_type=rain_type,
        p=1.0
    )
    
    return rain_transform(image=image)['image']

def apply_adaptive_sunflare(image, intensity='medium'):
    """Apply sun flare effect with adaptive parameters in multiple locations"""
    params = get_adaptive_sunflare_params(image.shape)
    result_image = image.copy()
    
    if intensity == 'light':
        angle_lower = 0.8  # More noticeable sun flare
        flare_count = 2  # Number of flares for light intensity
    elif intensity == 'medium':
        angle_lower = 0.7  # Moderately visible sun flare
        flare_count = 3  # Number of flares for medium intensity
    else:  # strong
        angle_lower = 0.6  # More pronounced effect
        flare_count = 4  # Number of flares for strong intensity
    
    # Multiple flare regions (top, middle, bottom sections)
    flare_regions = [
        (0, 0, 1.0, 0.4),    # Top section
        (0, 0.2, 1.0, 0.6),   # Upper-middle section
        (0, 0.4, 1.0, 0.8),   # Lower-middle section
        (0, 0.6, 1.0, 1.0)    # Bottom section
    ]
    
    # Apply multiple sun flares in different regions
    for i in range(min(flare_count, len(flare_regions))):
        sunflare_transform = A.RandomSunFlare(
            flare_roi=flare_regions[i],
            angle_lower=angle_lower,
            src_radius=params.get('src_radius', 50),  # Use larger radius
            p=0.7  # 70% chance per region
        )
        result_image = sunflare_transform(image=result_image)['image']
    
    return result_image

# --- Input and Output Paths ---
base_folder = r"C:\Users\Mahendra\OneDrive - TVM Signalling and Transportation Systems Pvt. Ltd\Projects\S4\Training_ML_Model\Training_Data02072025\imageaug-feature_yash\imageaug-feature_yash\images1"
input_img_folder = os.path.join(base_folder, "images")
input_label_folder = os.path.join(base_folder, "labels")

output_base = os.path.join(base_folder, "combined_aug")
output_img = os.path.join(output_base, "images")
output_lbl = os.path.join(output_base, "labels")
output_highlighted = os.path.join(output_base, "highlighted")

os.makedirs(output_img, exist_ok=True)
os.makedirs(output_lbl, exist_ok=True)
os.makedirs(output_highlighted, exist_ok=True)

# --- Class ID remapping ---
horizontal_flip_mapping = {'0': '1', '1': '0', '2': '3', '3': '2'}
rotation_90_270_mapping = {'0': '3', '1': '2', '2': '1', '3': '0'}

# --- Albumentations Setup ---
bbox_params = A.BboxParams(format='yolo', label_fields=['category_id'])

transformations = {
    "hflip": A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
    "rotate20": A.Compose([A.Rotate(limit=(20, 20), p=1.0)], bbox_params=bbox_params),
    "rotate45": A.Compose([A.Rotate(limit=(45, 45), p=1.0)], bbox_params=bbox_params),
    "rotate-40": A.Compose([A.Rotate(limit=(-40, -40), p=1.0)], bbox_params=bbox_params),
    "rotate-15": A.Compose([A.Rotate(limit=(-15, -15), p=1.0)], bbox_params=bbox_params),
    "aug_256": A.Compose([
        A.Resize(256, 256),
        A.Cutout(num_holes=5, max_h_size=20, max_w_size=20, fill_value=0, p=1.0)
    ], bbox_params=bbox_params),
    "aug_512": A.Compose([
        A.Resize(512, 512),
        A.Cutout(num_holes=10, max_h_size=35, max_w_size=35, fill_value=0, p=1.0)
    ], bbox_params=bbox_params),
    "noisy": None,  # handled manually
    "brightness": A.Compose([
        A.ColorJitter(brightness=1.4, contrast=0, saturation=0, hue=0, p=1.0)
    ], bbox_params=bbox_params),

    "contrast": A.Compose([
        A.ColorJitter(brightness=0, contrast=1.6, saturation=0, hue=0, p=1.0)
    ], bbox_params=bbox_params),

    "saturation": A.Compose([
        A.ColorJitter(brightness=0, contrast=0, saturation=1.5, hue=0, p=1.0)
    ], bbox_params=bbox_params),

    "hue": A.Compose([
        A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.4, p=1.0)
    ], bbox_params=bbox_params),

    "blur": A.Compose([
        A.Resize(1080, 1080),
        A.Blur(blur_limit=(3, 7), p=1.0),
        A.RandomSizedBBoxSafeCrop(height=800, width=800, erosion_rate=0.0, p=1.0)
    ], bbox_params=bbox_params),

    "defocus": A.Compose([
        A.Resize(1080, 1080),
        A.Defocus(radius=(3, 7), p=1.0),
        A.RandomSizedBBoxSafeCrop(height=800, width=800, erosion_rate=0.0, p=1.0)
    ], bbox_params=bbox_params),

    "glassblur": A.Compose([
        A.Resize(1080, 1080),
        A.GlassBlur(sigma=0.5, max_delta=2, iterations=2, p=1.0),
        A.RandomSizedBBoxSafeCrop(height=800, width=800, erosion_rate=0.0, p=1.0)
     ], bbox_params=bbox_params),

    # Weather and lighting augmentations (adaptive versions handled manually)
    "rain1": None,  # adaptive light rain - handled manually
    "rain2": None,  # adaptive heavy rain - handled manually
    "snow1": A.Compose([A.RandomSnow(brightness_coeff=2.0, p=1.0)], bbox_params=bbox_params),
    "snow2": A.Compose([A.RandomSnow(snow_point_upper=0.6, brightness_coeff=2.7, p=1.0)], bbox_params=bbox_params),
    "fog1": A.Compose([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.05, p=1.0)], bbox_params=bbox_params),
    "fog2": A.Compose([A.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.7, alpha_coef=0.15, p=1.0)], bbox_params=bbox_params),
    "sunflare1": None,  # adaptive light sunflare - handled manually
    "sunflare2": None,  # adaptive medium sunflare - handled manually
    "sunflare3": None,  # adaptive strong sunflare - handled manually
    "shadow1": A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=7, p=1.0)], bbox_params=bbox_params),
    "shadow2": A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=2, num_shadows_upper=6, shadow_dimension=10, p=1.0)], bbox_params=bbox_params),
    "shadow3": A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=3, num_shadows_upper=10, shadow_dimension=15, p=1.0)], bbox_params=bbox_params),
    "lightdrizzle": None,  # adaptive light drizzle + shadow - handled manually
    "heavyrain": None,  # adaptive heavy rain + shadow - handled manually
    "motionblur": A.Compose([A.MotionBlur(blur_limit=(3, 15), allow_shifted=True, p=1.0)], bbox_params=bbox_params),
    "brightnesscontrast": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=1.0)], bbox_params=bbox_params),
    "clahe": A.Compose([A.CLAHE(clip_limit=5.0, tile_grid_size=(8, 8), p=1.0)], bbox_params=bbox_params),
    "rgbshift": A.Compose([A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0)], bbox_params=bbox_params),
    "huesaturationvalue": A.Compose([A.HueSaturationValue(hue_shift_limit=35, sat_shift_limit=40, val_shift_limit=15, p=1.0)], bbox_params=bbox_params),
    "imagecompression": A.Compose([A.ImageCompression(quality_lower=5, quality_upper=20, p=1.0)], bbox_params=bbox_params),
    "colorjitter2": A.Compose([A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)], bbox_params=bbox_params),
}

# --- Process all images ---
for filename in os.listdir(input_img_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_img_folder, filename)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(input_label_folder, label_filename)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            continue

        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load YOLO labels ---
        bboxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, *box = parts
                        bboxes.append(list(map(float, box)))
                        class_ids.append(cls)
        if not bboxes:
            print(f"⚠️ Skipped: No labels for {filename}")
            continue

        # Save original image, labels, and highlighted version
        cv2.imwrite(os.path.join(output_img, filename), image)
        with open(os.path.join(output_lbl, label_filename), "w") as f:
            for cls, box in zip(class_ids, bboxes):
                f.write(f"{cls} {' '.join(map(str, box))}\n")

        orig_highlight = image.copy()
        for box in bboxes:
            x_center, y_center, bw, bh = box
            x_min = int((x_center - bw / 2) * width)
            y_min = int((y_center - bh / 2) * height)
            x_max = int((x_center + bw / 2) * width)
            y_max = int((y_center + bh / 2) * height)
            cv2.rectangle(orig_highlight, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_highlighted, filename), orig_highlight)

        alb_boxes = [{"bbox": box, "category_id": cls} for box, cls in zip(bboxes, class_ids)]

        # Apply augmentations
        for aug_name, transform in transformations.items():
            aug_img_name = f"{aug_name}_{filename}"
            aug_lbl_name = f"{os.path.splitext(aug_img_name)[0]}.txt"

            if aug_name == "noisy":
                aug_image = add_salt_pepper_noise(image_rgb)
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "rain1":
                aug_image = apply_adaptive_rain(image_rgb, rain_type='drizzle', intensity='light')
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "rain2":
                aug_image = apply_adaptive_rain(image_rgb, rain_type='heavy', intensity='heavy')
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "sunflare1":
                aug_image = apply_adaptive_sunflare(image_rgb, intensity='light')
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "sunflare2":
                aug_image = apply_adaptive_sunflare(image_rgb, intensity='medium')
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "sunflare3":
                aug_image = apply_adaptive_sunflare(image_rgb, intensity='strong')
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "lightdrizzle":
                # Apply adaptive light drizzle + shadow
                aug_image = apply_adaptive_rain(image_rgb, rain_type='drizzle', intensity='light')
                # Apply shadow on top of rain
                shadow_transform = A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=3, p=1.0)
                aug_image = shadow_transform(image=aug_image)['image']
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif aug_name == "heavyrain":
                # Apply adaptive heavy rain + shadow
                aug_image = apply_adaptive_rain(image_rgb, rain_type='heavy', intensity='heavy')
                # Apply shadow on top of rain
                shadow_transform = A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=1.0)
                aug_image = shadow_transform(image=aug_image)['image']
                aug_bboxes = bboxes
                aug_classes = class_ids
            elif transform is None:
                # Skip None transforms (already handled above)
                continue
            else:
                try:
                    transformed = transform(
                        image=image_rgb,
                        bboxes=[b["bbox"] for b in alb_boxes],
                        category_id=[b["category_id"] for b in alb_boxes]
                    )
                    aug_image = transformed["image"]
                    aug_bboxes = transformed["bboxes"]
                    aug_classes = transformed["category_id"]

                    if aug_name == "hflip":
                        aug_classes = [horizontal_flip_mapping.get(cls, cls) for cls in aug_classes]
                    elif "rotate" in aug_name:
                        try:
                            angle = int(aug_name.replace("rotate", ""))
                            if 90 < abs(angle) < 270:
                                aug_classes = [rotation_90_270_mapping.get(cls, cls) for cls in aug_classes]
                        except ValueError:
                            pass

                    if any(not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1) for (x, y, w, h) in aug_bboxes):
                        print(f"⚠️ Skipped: Out-of-bound bbox in {aug_img_name}")
                        continue

                except Exception as e:
                    print(f"❌ Augmentation error on {filename} with {aug_name}: {e}")
                    continue

            # Save image and labels
            cv2.imwrite(os.path.join(output_img, aug_img_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            with open(os.path.join(output_lbl, aug_lbl_name), "w") as f:
                for cls, bbox in zip(aug_classes, aug_bboxes):
                    f.write(f"{cls} {' '.join(map(str, bbox))}\n")

            # Save highlighted image
            highlighted = aug_image.copy()
            h, w = highlighted.shape[:2]
            for box in aug_bboxes:
                x_center, y_center, bw, bh = box
                x_min = int((x_center - bw / 2) * w)
                y_min = int((y_center - bh / 2) * h)
                x_max = int((x_center + bw / 2) * w)
                y_max = int((y_center + bh / 2) * h)
                cv2.rectangle(highlighted, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.imwrite(os.path.join(output_highlighted, aug_img_name), cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR))

print("✅ Done! All images processed with multiple augmentations.")
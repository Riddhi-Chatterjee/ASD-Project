from PIL import Image
import numpy as np


def load_image_by_pil(file_name, respect_exif=False):
    if isinstance(file_name, str):
        image = Image.open(file_name).convert('RGB')
    elif isinstance(file_name, bytes):
        import io
        image = Image.open(io.BytesIO(file_name)).convert('RGB')
    if respect_exif:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
    return image

def load_mask_by_pil(id_file_name, respect_exif=False):
    delimiter = "o_o"
    scene_instance_id = int(id_file_name.split(delimiter, 1)[0])
    file_name = id_file_name.split(delimiter, 1)[1]
    
    if isinstance(file_name, str):
        mask = Image.open(file_name)
    elif isinstance(file_name, bytes):
        import io
        mask = Image.open(io.BytesIO(file_name))
    if respect_exif:
        from PIL import ImageOps
        mask = ImageOps.exif_transpose(mask)
        
    mask_array = np.array(mask)

    # Create a binary mask (1 where scene_instance_id, 0 otherwise)
    binary_mask = np.where(mask_array == scene_instance_id, 1, 0)

    # Expand to 3 channels (R, G, B)
    # Convert binary mask to RGB format by replicating the mask across three channels
    rgb_mask = np.stack([binary_mask] * 3, axis=-1) * 255  # Multiply by 255 for 8-bit RGB

    # Convert back to PIL Image in RGB format
    rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8), mode='RGB')
    
    return rgb_mask


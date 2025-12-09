import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

### INSTURCTIONS
# If you downloaded the THINGS images from the official website and want to exactly replicate
# experimental conditions where images where shown at a specific resolution you can use this file.
# If you are generating stimulus sets for local benchmarking with the Gifford2022 and Papale2025 benchmarks,
# use this file to create 500x500 pixel images.
# Supply this file with the path to the originals.
# Then use the target path as the input path to get_local_stimuli.py.

def create_resized_database(source_path, target_path, target_size=500, quality=95):
    """
    Create a copy of the THINGS image database where alle images have the same resolution.
    
    Assumes all images are square, which should generally be the case for THINGS.
    
    Parameters:
    -----------
    source_path : str
        Path to the original THINGS database
    target_path : str
        Path where the resized database will be created
    target_size : int
        Target dimension (images will be target_size x target_size)
    quality : int
        JPEG quality for saving (1-100)
    """
    os.makedirs(target_path, exist_ok=True)
    
    categories = sorted([d for d in os.listdir(source_path) 
                        if os.path.isdir(os.path.join(source_path, d))])
    
    total_processed = 0
    
    for category in tqdm(categories, desc="Processing categories"):
        source_category_path = os.path.join(source_path, category)
        target_category_path = os.path.join(target_path, category)
        
        os.makedirs(target_category_path, exist_ok=True)
        
        image_files = [f for f in os.listdir(source_category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for img_file in image_files:
            source_img_path = os.path.join(source_category_path, img_file)
            target_img_path = os.path.join(target_category_path, img_file)
            
            with Image.open(source_img_path) as img:
                resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                resized.save(target_img_path, 'JPEG', quality=quality, subsampling=0)
                total_processed += 1

    print(f"Database resizing complete!")
    print(f"Total images processed: {total_processed:,}")
    print(f"Output location: {target_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mirror the THINGS image database changing image resolution to a common size."
    )
    parser.add_argument(
        "--things_image_dir", 
        type=str, 
        required=True,
        help="Path to the original THINGS database, it should end in 'object_images'"
    )
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True,
        help="Path where the resized database will be created, should end in 'object_images'"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=500, 
        help="Target dimension (THINGS images are all square). Default: 500x500 pixels"
    )
    parser.add_argument(
        "--quality", 
        type=int, 
        default=95, 
        help="JPEG quality for saving (1-100). Default: 95"
    )
    
    args = parser.parse_args()
    
    print(f"Source: {args.things_image_dir}")
    print(f"Target: {args.target_dir}")
    print(f"Target size: {args.size}x{args.size} pixels")
    print(f"Quality: {args.quality}")
    
    create_resized_database(
        args.things_image_dir, 
        args.target_dir, 
        target_size=args.size, 
        quality=args.quality
    )
    
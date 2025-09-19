#!/usr/bin/env python3
"""
Waterbirds Dataset Generation Script

This script creates the Waterbirds dataset for studying background bias in computer vision,
adapted from the original group_DRO implementation.

Usage:
    python generate_waterbirds.py --cub_dir ./CUB-200 --places_dir ./places365 --output_dir ./waterbirds_data

Original source: https://github.com/kohpangwei/group_DRO
"""

import os
import numpy as np
import random
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict


def crop_and_resize(source_img: Image.Image, target_img: Image.Image) -> Image.Image:
    """
    Make source_img exactly the same size as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.Resampling.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.Resampling.LANCZOS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.Resampling.LANCZOS)
    return source_resized


def combine_and_mask(img_new: Image.Image, mask: np.ndarray, img_black: Image.Image) -> Image.Image:
    """
    Combine img_new, mask, and image_black based on the mask

    Args:
        img_new: new (unmasked image)
        mask: binary mask of bird image
        img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined


def validate_paths(cub_dir: str, cub_seg_dir: str, places_dir: str) -> None:
    """Validate that all required data directories exist"""
    required_paths = [
        (cub_dir, "CUB dataset directory"),
        (cub_seg_dir, "CUB segmentation directory"),
        (places_dir, "Places365 dataset directory")
    ]
    
    for path, description in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{description} not found at: {path}")
    
    # Check for required files
    required_files = [
        (os.path.join(cub_dir, 'images.txt'), "CUB images.txt"),
        (os.path.join(cub_dir, 'train_test_split.txt'), "CUB train_test_split.txt"),
        (os.path.join(places_dir, 'categories_places365.txt'), "Places365 categories file")
    ]
    
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{description} not found at: {file_path}")


def setup_bird_labels(cub_dir: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Set up labels of waterbirds vs. landbirds"""
    images_path = os.path.join(cub_dir, 'images.txt')
    
    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id'
    )

    # We consider water birds = seabirds and waterfowl.
    species = np.unique([img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']])
    water_birds_list = [
        'Albatross', 'Auklet', 'Cormorant', 'Frigatebird', 'Fulmar', 'Gull', 'Jaeger',
        'Kittiwake', 'Pelican', 'Puffin', 'Tern', 'Gadwall', 'Grebe', 'Mallard',
        'Merganser', 'Guillemot', 'Pacific_Loon'
    ]

    water_birds = {}
    for species_name in species:
        water_birds[species_name] = 0
        for water_bird in water_birds_list:
            if water_bird.lower() in species_name:
                water_birds[species_name] = 1

    species_list = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']]
    df['y'] = [water_birds[species] for species in species_list]
    
    return df, water_birds


def setup_train_test_split(df: pd.DataFrame, cub_dir: str, val_frac: float = 0.2) -> pd.DataFrame:
    """Assign train/test/valid splits"""
    train_test_df = pd.read_csv(
        os.path.join(cub_dir, 'train_test_split.txt'),
        sep=" ",
        header=None,
        names=['img_id', 'split'],
        index_col='img_id'
    )

    df = df.join(train_test_df, on='img_id')
    test_ids = df.loc[df['split'] == 0].index
    train_ids = np.array(df.loc[df['split'] == 1].index)
    val_ids = np.random.choice(
        train_ids,
        size=int(np.round(val_frac * len(train_ids))),
        replace=False
    )

    df.loc[train_ids, 'split'] = 0  # train
    df.loc[val_ids, 'split'] = 1    # val
    df.loc[test_ids, 'split'] = 2   # test
    
    return df


def assign_confounders(df: pd.DataFrame, confounder_strength: float = 0.95) -> pd.DataFrame:
    """Assign confounders (place categories)"""
    # Confounders are set up as the following:
    # Y = 0, C = 0: confounder_strength
    # Y = 0, C = 1: 1 - confounder_strength
    # Y = 1, C = 0: 1 - confounder_strength
    # Y = 1, C = 1: confounder_strength

    df['place'] = 0
    train_ids = np.array(df.loc[df['split'] == 0].index)
    val_ids = np.array(df.loc[df['split'] == 1].index)
    test_ids = np.array(df.loc[df['split'] == 2].index)
    
    for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
        for y in (0, 1):
            if split_idx == 0:  # train
                if y == 0:
                    pos_fraction = 1 - confounder_strength
                else:
                    pos_fraction = confounder_strength
            else:
                pos_fraction = 0.5
            subset_df = df.loc[ids, :]
            y_ids = np.array((subset_df.loc[subset_df['y'] == y]).index)
            pos_place_ids = np.random.choice(
                y_ids,
                size=int(np.round(pos_fraction * len(y_ids))),
                replace=False
            )
            df.loc[pos_place_ids, 'place'] = 1
    
    return df


def assign_places(df: pd.DataFrame, places_dir: str, target_places: List[List[str]]) -> pd.DataFrame:
    """Assign places to train, val, and test set"""
    place_ids_df = pd.read_csv(
        os.path.join(places_dir, 'categories_places365.txt'),
        sep=" ",
        header=None,
        names=['place_name', 'place_id'],
        index_col='place_id'
    )

    target_place_ids = []

    for idx, target_place_list in enumerate(target_places):
        place_filenames = []

        for target_place in target_place_list:
            target_place_full = f'/{target_place[0]}/{target_place}'
            matching_places = place_ids_df['place_name'] == target_place_full
            
            if np.sum(matching_places) != 1:
                raise ValueError(f"Place {target_place_full} ERROR: {np.sum(matching_places)} matches")
            
            target_place_ids.append(place_ids_df.index[matching_places][0])
            print(f'Category {idx} {target_place_full} has id {target_place_ids[-1]}')

            # Read place filenames associated with target_place
            place_dir = os.path.join(places_dir, 'train', target_place)
            if not os.path.exists(place_dir):
                raise FileNotFoundError(f"Place directory not found: {place_dir}")
            
            place_filenames += [
                f'/{target_place}/{filename}' for filename in os.listdir(place_dir)
                if filename.endswith('.jpg')
            ]

        random.shuffle(place_filenames)

        # Assign each filename to an image
        indices = (df.loc[:, 'place'] == idx)
        if len(place_filenames) < np.sum(indices):
            raise ValueError(f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(indices)})")
        
        df.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

    return df


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """Print dataset statistics"""
    for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
        print(f"\n{split_label.upper()}:")
        split_df = df.loc[df['split'] == split, :]
        print(f"  Waterbirds are {np.mean(split_df['y']):.3f} of the examples")
        print(f"  y=0, c=0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n={np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}")
        print(f"  y=0, c=1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n={np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}")
        print(f"  y=1, c=0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n={np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}")
        print(f"  y=1, c=1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n={np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}")


def generate_waterbirds_dataset(
    cub_dir: str,
    cub_seg_dir: str, 
    places_dir: str,
    output_dir: str,
    dataset_name: str = 'waterbird_complete95_forest2water2',
    target_places: List[List[str]] = None,
    val_frac: float = 0.2,
    confounder_strength: float = 0.95,
    seed: int = 42
) -> None:
    """Generate the complete Waterbirds dataset"""
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    if target_places is None:
        target_places = [
            ['bamboo_forest', 'forest-broadleaf'],  # Land backgrounds
            ['ocean', 'lake-natural']              # Water backgrounds
        ]

    # Validate input paths
    validate_paths(cub_dir, cub_seg_dir, places_dir)

    print("Setting up bird labels...")
    df, water_birds = setup_bird_labels(cub_dir)

    print("Setting up train/test split...")
    df = setup_train_test_split(df, cub_dir, val_frac)

    print("Assigning confounders...")
    df = assign_confounders(df, confounder_strength)

    print("Assigning places...")
    df = assign_places(df, places_dir, target_places)

    print_dataset_statistics(df)

    # Create output directory with informative subdirectory
    output_subfolder = os.path.join(output_dir, dataset_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Save metadata
    metadata_path = os.path.join(output_subfolder, 'metadata.csv')
    df.to_csv(metadata_path)
    print(f"Saved metadata to: {metadata_path}")

    # Generate composite images
    print("Generating composite images...")
    for i in tqdm(df.index, desc="Processing images"):
        # Load bird image and segmentation
        img_path = os.path.join(cub_dir, 'images', df.loc[i, 'img_filename'])
        seg_path = os.path.join(cub_seg_dir, df.loc[i, 'img_filename'].replace('.jpg', '.png'))
        
        if not os.path.exists(img_path):
            print(f"Warning: Missing image {img_path}")
            continue
        if not os.path.exists(seg_path):
            print(f"Warning: Missing segmentation {seg_path}")
            continue

        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        place_path = os.path.join(places_dir, 'train', df.loc[i, 'place_filename'][1:])
        if not os.path.exists(place_path):
            print(f"Warning: Missing place {place_path}")
            continue
            
        place = Image.open(place_path).convert('RGB')

        # Create composite image
        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)

        # Save composite image
        output_path = os.path.join(output_subfolder, df.loc[i, 'img_filename'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_img.save(output_path)

    print(f"\nDataset generation completed!")
    print(f"Dataset saved to: {output_subfolder}")
    print(f"Total images processed: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Generate Waterbirds dataset for bias research")
    parser.add_argument("--cub_dir", type=str, required=True,
                       help="Path to CUB-200 dataset directory")
    parser.add_argument("--cub_seg_dir", type=str, default=None,
                       help="Path to CUB-200 segmentations directory (default: cub_dir + '_segmentations')")
    parser.add_argument("--places_dir", type=str, required=True,
                       help="Path to Places365 dataset directory")
    parser.add_argument("--output_dir", type=str, default="./waterbirds_data",
                       help="Output directory for Waterbirds dataset")
    parser.add_argument("--dataset_name", type=str, default="waterbird_complete95_forest2water2",
                       help="Name for the generated dataset")
    parser.add_argument("--val_frac", type=float, default=0.2,
                       help="Fraction of training data to use as validation")
    parser.add_argument("--confounder_strength", type=float, default=0.95,
                       help="Strength of the bias correlation (0.5=no bias, 1.0=perfect bias)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set default segmentation directory if not provided
    if args.cub_seg_dir is None:
        args.cub_seg_dir = args.cub_dir + "_segmentations"

    print("Generating Waterbirds Dataset")
    print("=" * 40)
    print(f"CUB directory: {args.cub_dir}")
    print(f"CUB segmentation directory: {args.cub_seg_dir}")
    print(f"Places365 directory: {args.places_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Validation fraction: {args.val_frac}")
    print(f"Confounder strength: {args.confounder_strength}")
    print(f"Random seed: {args.seed}")
    print("=" * 40)

    generate_waterbirds_dataset(
        cub_dir=args.cub_dir,
        cub_seg_dir=args.cub_seg_dir,
        places_dir=args.places_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        val_frac=args.val_frac,
        confounder_strength=args.confounder_strength,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
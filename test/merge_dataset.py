import os
import json
import shutil
from tqdm import tqdm

def merge_coco_datasets(dataset_paths, output_dir):
    """
    Merges multiple COCO datasets into one.

    :param dataset_paths: A list of dictionaries, where each dict contains 'images_dir' and 'ann_file'.
    :param output_dir: Path to the directory where the merged dataset will be saved.
    """
    
    # Create output directories
    merged_images_dir = os.path.join(output_dir, 'images')
    merged_anns_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(merged_images_dir, exist_ok=True)
    os.makedirs(merged_anns_dir, exist_ok=True)

    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    max_img_id = 0
    max_ann_id = 0
    category_map = {} # To handle unique categories

    print("Starting dataset merge...")

    for dataset in dataset_paths:
        print(f"Processing annotation file: {dataset['ann_file']}")
        with open(dataset['ann_file'], 'r') as f:
            data = json.load(f)

        # --- Merge Categories ---
        if not merged_data['categories']:
             if 'categories' in data:
                merged_data['categories'] = data['categories']
                for cat in data['categories']:
                    category_map[cat['name']] = cat['id']
        else:
            if 'categories' in data:
                for cat in data['categories']:
                    if cat['name'] not in category_map:
                        # This part is simplified assuming single class 'smoke' with same id.
                        # For multi-class, more complex mapping would be needed.
                        print(f"Warning: new category '{cat['name']}' found. Merging logic might need adjustment for multi-class.")
                        # merged_data['categories'].append(cat)
                        # category_map[cat['name']] = cat['id']


        # --- Merge Images and Annotations ---
        img_id_mapping = {}
        for img in tqdm(data['images'], desc=f"  - Images from {os.path.basename(dataset['ann_file'])}"):
            original_img_id = img['id']
            new_img_id = original_img_id + max_img_id
            img_id_mapping[original_img_id] = new_img_id
            
            img['id'] = new_img_id
            merged_data['images'].append(img)

            # Create symlink for the image file
            src_img_path = os.path.join(dataset['images_dir'], img['file_name'])
            dst_img_path = os.path.join(merged_images_dir, img['file_name'])

            if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
                 os.symlink(os.path.abspath(src_img_path), dst_img_path)


        if 'annotations' in data:
            for ann in tqdm(data['annotations'], desc=f"  - Annotations from {os.path.basename(dataset['ann_file'])}"):
                original_img_id = ann['image_id']
                if original_img_id in img_id_mapping:
                    ann['image_id'] = img_id_mapping[original_img_id]
                    ann['id'] += max_ann_id
                    merged_data['annotations'].append(ann)
        
        # Update max IDs for the next dataset
        current_max_img_id = max([img['id'] for img in data['images']] or [0])
        current_max_ann_id = max([ann['id'] for ann in data.get('annotations', [])] or [0])
        max_img_id += current_max_img_id
        max_ann_id += current_max_ann_id

    # Save merged annotation file
    output_ann_file = os.path.join(merged_anns_dir, 'merged.json')
    print(f"\nSaving merged annotations to {output_ann_file}")
    with open(output_ann_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print("Merge complete!")
    return output_ann_file


def organize_for_dino(merged_dir, split):
    """ Renames and organizes merged files into DINO-compatible structure. """
    src_json = os.path.join(merged_dir, 'annotations', 'merged.json')
    dst_json_dir = os.path.join(merged_dir, 'annotations')
    dst_img_dir = os.path.join(merged_dir, f'{split}2017')
    
    # Rename merged.json to instances_train2017.json or instances_val2017.json
    final_json_path = os.path.join(dst_json_dir, f'instances_{split}2017.json')
    shutil.move(src_json, final_json_path)
    
    # Rename the 'images' directory to 'train2017' or 'val2017'
    shutil.move(os.path.join(merged_dir, 'images'), dst_img_dir)
    
    print(f"Organized for DINO: '{split}' set is ready.")


def main():
    # --- !! IMPORTANT: PLEASE VERIFY THESE PATHS !! ---
    base_dir = 'datasets/data'
    
    # Define your source datasets
    datasets_to_merge_train = [
        {
            "images_dir": os.path.join(base_dir, 'FASDD_CV/images'),
            "ann_file": os.path.join(base_dir, 'FASDD_CV/annotations/COCO_CV/Annotations/train.json')
        },
        {
            "images_dir": os.path.join(base_dir, 'FASDD_UAV/images'),
            "ann_file": os.path.join(base_dir, 'FASDD_UAV/annotations/COCO_UAV/Annotations/train.json')
        }
    ]
    
    datasets_to_merge_val = [
        {
            "images_dir": os.path.join(base_dir, 'FASDD_CV/images'),
            "ann_file": os.path.join(base_dir, 'FASDD_CV/annotations/COCO_CV/Annotations/val.json')
        },
        {
            "images_dir": os.path.join(base_dir, 'FASDD_UAV/images'),
            "ann_file": os.path.join(base_dir, 'FASDD_UAV/annotations/COCO_UAV/Annotations/val.json')
        }
    ]
    
    # Define the final output directory for the combined dataset
    final_output_dir = 'coco_smoke_merged'
    
    # --- Execution ---
    # Create the final directory
    if os.path.exists(final_output_dir):
        print(f"Output directory '{final_output_dir}' already exists. Removing it.")
        shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir)

    # Process training data
    print("--- Merging Training Sets ---")
    merged_train_dir = os.path.join(final_output_dir, 'train_temp')
    merge_coco_datasets(datasets_to_merge_train, merged_train_dir)
    organize_for_dino(merged_train_dir, 'train')
    # Move organized files to final destination
    shutil.move(os.path.join(merged_train_dir, 'annotations'), os.path.join(final_output_dir, 'annotations'))
    shutil.move(os.path.join(merged_train_dir, 'train2017'), os.path.join(final_output_dir, 'train2017'))
    os.rmdir(merged_train_dir)


    # Process validation data
    print("\n--- Merging Validation Sets ---")
    merged_val_dir = os.path.join(final_output_dir, 'val_temp')
    merge_coco_datasets(datasets_to_merge_val, merged_val_dir)
    organize_for_dino(merged_val_dir, 'val')
    # Move organized files to final destination
    shutil.move(os.path.join(merged_val_dir, 'val2017'), os.path.join(final_output_dir, 'val2017'))
    # The annotations directory already exists, so we just move the val json file
    shutil.move(
        os.path.join(merged_val_dir, 'annotations', 'instances_val2017.json'), 
        os.path.join(final_output_dir, 'annotations', 'instances_val2017.json')
    )
    shutil.rmtree(merged_val_dir)

    print(f"\n\nSUCCESS! Your merged dataset is ready at: {os.path.abspath(final_output_dir)}")
    print("You can now use this path as 'data_root' in your DINO config file.")


if __name__ == '__main__':
    main()

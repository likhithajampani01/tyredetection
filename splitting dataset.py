import os
import shutil


def reorganize_dataset(current_base_folder, fixed_base_folder):
    """
    Move images from the wrong structure to the correct train/val/test split structure.

    Expected input:
    current_base_folder/defective/train, val, test
    current_base_folder/good/train, val, test

    Desired output:
    fixed_base_folder/train/defective
    fixed_base_folder/val/defective
    fixed_base_folder/test/defective
    and same for 'good'.
    """

    classes = ['defective', 'good']
    splits = ['train', 'val', 'test']

    for class_name in classes:
        for split in splits:
            source_folder = os.path.join(current_base_folder, class_name, split)
            target_folder = os.path.join(fixed_base_folder, split, class_name)

            os.makedirs(target_folder, exist_ok=True)

            if os.path.exists(source_folder):
                for img_file in os.listdir(source_folder):
                    src = os.path.join(source_folder, img_file)
                    dst = os.path.join(target_folder, img_file)

                    if os.path.isfile(src):
                        shutil.move(src, dst)

    print("✅ Dataset structure reorganized successfully!")


if __name__ == '__main__':
    current_base_folder = r'C:\Users\HP\PycharmProjects\tyredetection\splittingdata'
    fixed_base_folder = r'C:\Users\HP\PycharmProjects\tyredetection\fixed_splittingdata'

    reorganize_dataset(current_base_folder, fixed_base_folder)

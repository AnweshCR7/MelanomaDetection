import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)


if __name__ == '__main__':
    input_folder = ""
    output_folder = ""
    n_jobs = 12

    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=n_jobs)(
        delayed(resize_image)(
            image,
            output_folder,
            (512, 512)
        ) for image in tqdm(images)
    )

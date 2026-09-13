import os
import cv2
import numpy as np
import natsort
import uiqm_plip_pro_0405 as uiqm


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in {
        '.png', '.jpg', '.jpeg', '.bmp'
    }


if __name__ == '__main__':
    folder = 'test results folder here'
    print(f'평가 대상 폴더: {folder}')

    files = [
        file for file in natsort.natsorted(os.listdir(folder))
        if is_image_file(file)
    ]

    uiqm_scores = []

    for i, file in enumerate(files, start=1):
        print(f'[{i}/{len(files)}] {file}')

        image_path = os.path.join(folder, file)

        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        score = uiqm.getUIQM(img)[1]
        uiqm_scores.append(score)

    avg_uiqm = np.mean(uiqm_scores)

    print(f'\n평균 UIQM: {avg_uiqm:.4f}')

    output_path = os.path.join(folder, 'uiqm_metrics.txt')

    with open(output_path, 'a') as f:
        folder_name = os.path.basename(folder.rstrip('/\\'))
        f.write(f'Folder:{folder_name}: uiqm={avg_uiqm:.4f}\n')

    print(f'결과 저장됨 → {output_path}')
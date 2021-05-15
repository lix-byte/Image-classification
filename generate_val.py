import os, random, shutil

def moveFile(fileDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    picknumber = int(filenumber * ratio)
    sample = random.sample(pathDir, picknumber)
    for name in sample:
        shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
    return


if __name__ == '__main__':
    ori_path = r'./Brain Tumor Classification/train'
    split_Dir = r'./Brain Tumor Classification/val'
    ratio = 0.2  # Draw ratio
    for firstPath in os.listdir(ori_path):
        fileDir = os.path.join(ori_path, firstPath)
        tarDir = os.path.join(split_Dir, firstPath)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        moveFile(fileDir)

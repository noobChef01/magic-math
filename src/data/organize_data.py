'''
Organize train and test files based on Imagenet data folder structure. 
'''


import os
from os.path import join
import uuid
import pickle
import argparse
import numpy as np
from numpy.core.fromnumeric import var
from tqdm import tqdm
from skimage import io


def read_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True,
                    help="Path to pickled train files.")
    ap.add_argument("--test_path", required=True,
                    help="Path to pickled test files.")
    ap.add_argument("--out_path", required=True,
                    help="Path to dump organized data.")
    args = vars(ap.parse_args())
    return args


def load_pickle(path_to_dir):
    result = []
    for file in tqdm(os.listdir(path_to_dir)):
        with open(join(path_to_dir, file), "rb") as file:
            equations = pickle.load(file, encoding="latin-1")
            result.extend(equations)
    return result


def save_image(data, out_path, train=True):
    if train:
        path = join(out_path, "train")
        os.makedirs(path, exist_ok=True)
    else:
        path = join(out_path, "test")
        os.makedirs(path, exist_ok=True)

    for equation in tqdm(data):
        for elem in equation:
            image = np.asarray(elem["features"], dtype="uint8")
            label = elem["label"]
            os.makedirs(join(path, label), exist_ok=True)
            fname = join(path, label, "{}.png".format(uuid.uuid4()))
            io.imsave(fname, image)


if __name__ == "__main__":
    args = read_args()
    train = load_pickle(args["train_path"])
    test = load_pickle(args["test_path"])
    save_image(train, args["out_path"])
    save_image(test, args["out_path"], train=False)

import os
import random
import io
import numpy as np
from PIL import Image


class BlackBox():

    def __init__(self, shredded_path, orig_path):
        self.shredded_image = Image.open(shredded_path)
        self.original_image = Image.open(orig_path)
        self.blocks = self.create_blocks()
        self.origin_matrix = self.PIL2array(self.original_image)
        self.np_matrix = self.PIL2array(self.shredded_image)

    def PIL2array(self, img):
        return np.array(img.getdata(),
                        np.uint8).reshape(img.size[1], img.size[0], 1)

    def array2PIL(self, arr, size):
        mode = 'L'
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
        if len(arr[0]) == 3:
            arr = np.c_[arr, 255*np.ones((len(arr), 1), np.uint8)]
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

    def create_blocks(self):
        blocks = []
        for i in range(1, 129):
            blocks.append(list(range((i-1)*5, i*5)))
        return blocks

    def swap(self, indexes, matrix):
        permutation = []
        for i in indexes:
            permutation.extend(self.blocks[i])
        return matrix[:, permutation]

    def evaluate_solution(self, permutation):
        if len(permutation) != len(self.blocks):
             raise Exception(
                 "Size of permutation list is wrong. It should be {0}".format(len(self.blocks)))

        np_matrix = self.swap(permutation, self.np_matrix)
        return np.sum(np.abs(np_matrix-self.origin_matrix))

    def show_solution(self, permutation, record=None):
        if not isinstance(permutation, list):
            raise Exception("You should provide a permutation list")
        np_matrix = self.PIL2array(self.shredded_image)
        np_matrix = self.swap(permutation, np_matrix)
        new_image = self.array2PIL(np_matrix, self.original_image.size)
        if record is None:
            new_image.show()
        else:
            new_image.save(record)

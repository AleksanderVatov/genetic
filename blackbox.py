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
        'Converts PIL img to numpy array.'
        return np.array(img.getdata(),
                        np.uint8).reshape(img.size[1], img.size[0], 1)

    def array2PIL(self, array, size):
        'Converts numpy array to PIL img.'
        mode = 'L'
        dim = array.shape
        array = array.reshape(dim[0] * dim[1], dim[2])

        if len(array[0]) == 3:
            # Concatenation along the second axis ie. [[a[0], 255], [a[1], 255], ...].
            array = np.c_[array, 255 * np.ones((len(array), 1), np.uint8)]
        
        return Image.frombuffer(mode, size, array.tostring(), 'raw', mode, 0, 1)


    def create_blocks(self):
        'Initializes blocks by increaments of 5.'
        blocks = []
        for i in range(0, 128):
            blocks.append(list(range(i*5, (i+1)*5)))

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

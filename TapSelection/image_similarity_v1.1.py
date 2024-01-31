import pdqhash
import cv2
import os
import numpy as np

def compute_pdq_hash(image):
    hash_vector, quality = pdqhash.compute(image)

    # Get all the rotations and flips in one pass.
    # hash_vectors is a list of vectors in the following order
    # - Original
    # - Rotated 90 degrees
    # - Rotated 180 degrees
    # - Rotated 270 degrees
    # - Flipped vertically
    # - Flipped horizontally
    # - Rotated 90 degrees and flipped vertically
    # - Rotated 90 degrees and flipped horizontally
    hash_vectors, quality = pdqhash.compute_dihedral(image)

    # Get the floating point values of the hash.
    hash_vector_float, quality = pdqhash.compute_float(image)
    
    return hash_vector_float

image1 = cv2.imread("./oc_2b.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image2 = cv2.imread("./oc_4.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

emb1 = compute_pdq_hash(image1)
emb2 = compute_pdq_hash(image2)

dist = np.square(np.subtract(emb1,emb2)).mean()
#dist = np.count_nonzero(emb1!=emb2)

print("Dist: "+str(dist))
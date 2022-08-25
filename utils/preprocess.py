import os
import pickle
import re
import numpy as np

def test(vlen):
    num_clips = int(min(np.ceil(vlen / 5), 10))
    if num_clips <= 1:
        return [0]
    elif num_clips < 6:
        ds = 9 // (num_clips - 1)
        seq_idx = [x*ds for x in range(1, num_clips-1)]
        return [0] + seq_idx + [9]
    elif num_clips == 6:
        return [0, 2, 4, 6, 8, 9]
    elif num_clips == 7:
        return [0, 1, 3, 5, 6, 8, 9]
    elif num_clips == 8:
        return [0, 1, 2, 4, 6, 7, 8, 9]
    elif num_clips >= 9:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in range(1,100):
    print(test(i))
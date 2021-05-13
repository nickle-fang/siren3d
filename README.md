# Siren3d
--------
    This is the repo testing siren.

## utils
    provide some utils to preprocess the data.
    + data_file_maker.py
        Given the data file address, the file name file for training and testing will be generated with given numver (random select)
        + **Input arg:**
            + image_addr : base addr containing rgb and depth image files dir
            + file_addr : where to store the tran and test txt file
            + tranin_number : Select how many images to train (The remains are for testing) 
        + **Usage**: python utils/data_file_maker.py -i /home/tonyhan/code/SirenPBA/data/train -o /home/tonyhan/code/siren3d/data -n 2
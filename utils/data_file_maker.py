import os 
import argparse
import random
import sys

dataset_name = 'nyu'

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--image_addr',help='Storing images address')
    parser.add_argument('-o','--output_addr',help='txt file writing address')
    parser.add_argument('-n','--train_number',help='Traing images number')

    args = parser.parse_args()

    # read images
    base_addr = os.path.expanduser(args.image_addr)
    rgb_addr = base_addr + '/rgb/'
    rgb_path_list = os.listdir(rgb_addr)
    if len(rgb_path_list) == 0:
        raise(RuntimeError("Detecting no rgb images"))

    print("<<<<<<<<<<<<<<<<<<<<<<<")
    print("Read images from {} ,Containing {} rgb images".format(base_addr,len(rgb_path_list)))
    print("<<<<<<<<<<<<<<<<<<<<<<<")

    train_number = int(args.train_number)
    


    if (train_number > len(rgb_path_list)):
        print('!!!!! train numver is {},it must smaller than all images number'.format(train_number))
        sys.exit()


    random_number = random.sample(range(0,len(rgb_path_list)),train_number)

    train_filename = os.path.expanduser(args.output_addr)+'/' + dataset_name + '_train.txt'
    test_filename = os.path.expanduser(args.output_addr)+'/' + dataset_name + '_test.txt'

    train_name = []
    test_name = []
    for i in range(len(rgb_path_list)):
        name = base_addr + '/%s/'+str(i)+'.png'
        if i in random_number:
            train_name.append(name)
        else:
            test_name.append(name)

    train_file = open(train_filename,'w+')
    test_file = open(test_filename,'w+')
    for i in train_name:
        train_file.write(i+ '\n')
    train_file.close()

    for j in test_name:
        test_file.write(j+ '\n')
    test_file.close()

    print("train_filename is {}".format(train_filename))
    print("test_filename is {}".format(test_filename))
    print("Out put txt files done")    
    print("<<<<<<<<<<<<<<<<<<<<<<<")
            
import numpy as np
import os, glob

# from NYUv2PyToolBox.nyuv2.raw import extract
from NYUv2PyToolBox.NYUv2PyToolBox.nyuv2.raw import load
from NYUv2PyToolBox.NYUv2PyToolBox.nyuv2.raw import extract
# from NYUv2PyToolBox.
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
from PIL import Image


# .mnt.Data.dataset.NYU_v2_depth.raw.single_file.NYUv2PyToolBox
def import_files(path):
    rgb_file_list = []
    depth_file_list = []
    a_file_list = []
    all_file_list = []
    all_file_list.extend(glob.glob(os.path.join(path, '*')))
    rgb_file_list.extend(glob.glob(os.path.join(path, 'r-*')))
    depth_file_list.extend(glob.glob(os.path.join(path, 'd-*')))
    a_file_list.extend(glob.glob(os.path.join(path, 'a-*')))
    # print("rgb_file_list",len(rgb_file_list))
    # print("depth_file_list",len(depth_file_list))
    # print("depth_file_list",len(a_file_list))
    # print("all_file_list",len(all_file_list))
    # print("Added#",len(rgb_file_list) +len(depth_file_list)+ len(a_file_list) )

    rgb_file_list.sort()
    depth_file_list.sort()



    depth_img_names, color_img_names = [], []

    for i in rgb_file_list:
        base_name = os.path.basename(i)
        # print(base_name)
        color_img_names.append(base_name)

    for ii in depth_file_list:
        base_name_c = os.path.basename(ii)
        depth_img_names.append(base_name_c)

    return  depth_img_names, color_img_names


def ckeck_file_exsist(frames,path):
    for i, curr_frame in enumerate(frames):

        ck_image = os.path.isfile(os.path.join(path, curr_frame[0]))
        ck_depth = os.path.isfile(os.path.join(path, curr_frame[1]))
        assert ck_image == True
        assert ck_depth == True


def process(frames, path):
    do_save = False
    do_plot_imshow = False

    max_fn = 4
    all_rgb, all_depth  = [], []




    for i, curr_frame in enumerate(frames):

        c_depth = load.load_depth_image(os.path.join(path, curr_frame[0]))
        c_img = load.load_color_image(os.path.join(path, curr_frame[1]))
        assert c_img.size == c_depth.size

        depth_uint8 = np.array(c_depth).astype(np.uint8)
        img_uint8 = np.array(c_img).astype(np.uint8)



        depth_ch = (depth_uint8 / 255.0)
        img_ch = (img_uint8 / 255.0)


        all_rgb.append([img_ch])
        all_depth.append([depth_ch])

        if do_plot_imshow:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.imshow(img_ch)
            ax2.imshow(depth_ch)

            ax1.title.set_text('Original_RGB_image')
            ax2.title.set_text('Depth - Saved in unit8')

            plt.show()


        if do_save:
            plt.imshow(c_depth)
            name = str(str(i) + "test_depth.png")
            from keras.preprocessing.image import save_img
            from keras.preprocessing.image import img_to_array

            img_array = img_to_array(c_depth)
            save_img(name, img_array)


            plt.imshow(c_img)
            name = str(str(i) + "test_rgb.jpg")
            plt.savefig(name)


    st_depth = np.vstack(all_depth)
    st_img= np.vstack(all_rgb)


    return st_img , st_depth



def extracting_zip(depth_img_names, color_img_names):
    zip_path = '/home/chna1572/workspace/depth_estimation/xtract/Datasets/nyu_v2_raw/test/basements.zip'
    frames = extract.synchronise_frames(depth_img_names, color_img_names)

    # if len(frames[0]) != len(frames[1]):
    #     print(len(frames[0]), len(frames[1]))

    # assert len(frames[0]) == len(frames[1])

    return frames

def get_file_location(path):
    import os
    list_dataset = []
    max = 3
    list_dir = os.listdir(path)
    for i, list_fld in enumerate (list_dir):
        curr = os.path.join(path, list_fld)
        list_dataset.append(curr)

        if i == max:
            break

    list_dataset.sort()

    print("MAXING DATASET", len(list_dataset))
    return list_dataset


def main_extract ():
    all_img, all_depth = [], []

    # path = '/mnt/Data/dataset/NYU_v2_depth/raw/single_file/testing_raw_dataset/basements/basement_0001a'
    path = '/mnt/Data/dataset/NYU_v2_depth/raw/single_file/nyu_depth_v2_raw'
    list_dataset = get_file_location(path)

    for i, dir in enumerate(list_dataset):



        depth_img_names, color_img_names = import_files(dir)
        frames, error_flag = extracting_zip(depth_img_names, color_img_names)
        # print(i,error_flag,  dir, os.path.isdir(dir))
        if error_flag== False:
            try:
                # ckeck_file_exsist(frames,dir)
                x_imgae, y_depth = process(frames, dir)
                print(np.shape(x_imgae), np.shape(y_depth))
                er_flage = True



            except:
                er_flage = False
                print("---ERROR",i, error_flag, dir, os.path.isdir(dir))



            if er_flage == True:
                print("SHAPE:", np.shape(x_imgae))
                all_img.append(x_imgae)
                all_depth.append(y_depth)






    x_patch = np.vstack(all_img)
    y_patch = np.vstack(all_depth)

    print("\n -- \n",np.shape(x_patch), np.shape(y_patch))



    return x_imgae, y_depth


















if __name__ == '__main__':
    main_extract()




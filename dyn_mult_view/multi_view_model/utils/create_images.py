import Image
import cPickle
import numpy as np
import matplotlib.pyplot as plt

def make_row(data, num_cols):
    row = []
    for c in range(num_cols):
        im = data[c]
        if im.shape[-1] == 1:
            cmap = plt.cm.get_cmap()
            # if renormalize:
            #     distrib[b] /= (np.max(distrib[b]) + 1e-6)
            im = cmap(np.squeeze(im))[:, :, :3]
        im = (im * 255).astype(np.uint8)

        im = np.pad(im,[(5,5), (5,5),(0,0)], 'constant', constant_values=[(255,255), (255,255),(0,0)])
        row.append(im)
    row = np.concatenate(row,axis=1)
    return row

class ImageMethod():
    def __init__(self, file):
        dict = cPickle.load(open(file, "rb"))
        self.image0 = dict['image0']
        self.image0_mask0 = dict['image0_mask0']
        self.image0_mask1 = dict['image0_mask1']
        self.image1 = dict['image1']
        self.image1_only0 = dict['image1_only0']
        self.image1_only1 = dict['image1_only1']
        self.image1_mask0 = dict['image1_mask0']
        self.image1_mask1 = dict['image1_mask1']
        self.depth0 = dict['depth0']
        self.depth1 = dict['depth1']
        self.depth1_only0 = dict['depth1_only0']
        self.depth1_only1 = dict['depth1_only1']
        self.gen_image1 = dict['gen_image1']
        self.gen_image1_only0 = dict['gen_image1_only0']
        self.gen_image1_only1 = dict['gen_image1_only1']
        self.gen_image1_mask0 = dict['gen_image1_mask0']
        self.gen_image1_mask1 = dict['gen_image1_mask1']
        self.gen_depth1 = dict['gen_depth1']
        self.gen_depth1_only0 = dict['gen_depth1_only0']
        self.gen_depth1_only1 = dict['gen_depth1_only1']


def create_images():

    imrows = []

    data1 = ImageMethod('/home/frederik/Documents/catkin_ws/src/dynamic_multiview_3d/tensorflowdata/multi_obj/fully_conv/modeldata/imgdata.pkl')
    data2 = ImageMethod('/home/frederik/Documents/catkin_ws/src/dynamic_multiview_3d/tensorflowdata/multi_obj/col_d_masks_combimg_sepimage/modeldata/imgdata.pkl')

    num_cols = 10

    imrows.append(make_row(data1.image0, num_cols))
    imrows.append(make_row(data1.image0_mask0, num_cols))
    imrows.append(make_row(data1.image0_mask1, num_cols))

    imrows.append(make_row(data1.gen_image1, num_cols))
    imrows.append(make_row(data1.gen_depth1, num_cols))

    imrows.append(make_row(data2.gen_image1, num_cols))
    imrows.append(make_row(data2.gen_depth1, num_cols))

    imrows.append(make_row(data1.image1, num_cols))
    imrows.append(make_row(data1.depth1, num_cols))

    image = np.concatenate(imrows, axis=0)

    file = '/home/frederik/Documents/catkin_ws/src/dynamic_multiview_3d/tensorflowdata/multi_obj/out_images/comp1.png'
    Image.fromarray(image).save(file)

if __name__ == '__main__':
    create_images()

import cv2
import numpy as np


def set_ker(degree):
    kernel_1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # 0 degree
    kernel_2 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])  # 45 degree
    kernel_3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # 90 degree
    kernel_4 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])  # 135 degree
    
    ker = {'0': kernel_1, '45': kernel_2, '90': kernel_3, '135': kernel_4}
    return ker[degree]


class edge_extraction:
    def __init__(self, image_path, output_path, filter_mode,threshold):
        self.image=cv2.imread(image_path, 0)
        self.output_path = output_path
        self.rows, self.cols = self.image.shape
        self.change_image = np.zeros([self.rows-1,self.cols-1])
        self.filter_mode = filter_mode
        self.kernel = set_ker(filter_mode)
        self.r = self.kernel.shape[0]//2
        self.threshold=threshold
    def Sobel_filter(self, index_x, index_y):
        gradient=(self.image[index_x-self.r:index_x+self.r+1,index_y-self.r:index_y+self.r+1]\
            *self.kernel).sum(0).sum(0)
        self.change_image[index_x-1,index_y-1]=255 if np.abs(gradient)>self.threshold else 0


    def Conv_func(self):
        for i in range(self.r, self.rows-self.r):
            for j in range(self.r, self.cols-self.r):
                self.Sobel_filter(i, j)
        cv2.imwrite(self.output_path, self.change_image)
        return self.image


if __name__ == "__main__":
    mode = input("Where direction to filter the image?[0? 45? 90? 135?]")
    out_path = "./output/{}_filter.jpg".format(mode)
    image_path = "./Wirebond.tif"
    threshold=0
    image=edge_extraction(image_path, out_path, mode,threshold)
    image.Conv_func()

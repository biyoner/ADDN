from img_utiles import *
import numpy as np
import random

class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset

    def load_npy_data(self,flag):
        imgs_trn = np.load(self.dataset + "/img_trn.npy")
        msks_trn = np.load(self.dataset + "/msk_trn.npy")
        imgs_val = np.load(self.dataset + "/img_val.npy")
        msks_val = np.load(self.dataset + "/msk_val.npy")
        imgs_tst = np.load(self.dataset + "/img_tst.npy")
        if flag == "train":
            return imgs_trn,msks_trn
        elif flag == "val":
            return imgs_val,msks_val
        elif flag == "submit":
            return imgs_tst

    def load_batch(self, batch_size=1):
        imgs_trn = np.load(self.dataset + "/new_img_trn.npy")
        msks_trn = np.load(self.dataset + "/new_msk_trn.npy")
        self.n_batches = int(imgs_trn.shape[0] / batch_size)
        for i in range(self.n_batches - 1):
            imgs =imgs_trn[i * batch_size:(i + 1) * batch_size]
            labels = msks_trn[i * batch_size:(i + 1) * batch_size]
            yield imgs, labels

    def load_img(self, batch_size=1):
        imgs_val = np.load(self.dataset + "/img_trn.npy")
        msks_val = np.load(self.dataset + "/msk_trn.npy")
        batch_images = random.sample(range(imgs_val.shape[0]), batch_size)
        return imgs_val[batch_images],msks_val[batch_images]


# if __name__ == "__main__":
#     data = DataLoader("data")
#     val_1,val2 = data.load_npy_data('val')
#     print val_1.shape
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
import tifffile as tiff
import cv2
from keras.layers import Concatenate

def get_data_montage(imgs_path, msks_path, nb_rows, nb_cols, rng):
    '''Reads the images and masks and arranges them in a montage for sampling in training.'''
    imgs, msks = tiff.imread(imgs_path), tiff.imread(msks_path) / 255
    montage_imgs = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.float32)
    montage_msks = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.int8)
    idxs = np.arange(imgs.shape[0])
    rng.shuffle(idxs)
    idxs = iter(idxs)
    for y0 in range(0, montage_imgs.shape[0], imgs.shape[1]):
        for x0 in range(0, montage_imgs.shape[1], imgs.shape[2]):
            y1, x1 = y0 + imgs.shape[1], x0 + imgs.shape[2]
            idx = next(idxs)
            montage_imgs[y0:y1, x0:x1] = imgs[idx]
            montage_msks[y0:y1, x0:x1] = msks[idx]
    return montage_imgs, montage_msks
def load_montage_data(imgs_trn,msks_trn,row_trn,imgs_val, msks_val,col_trn,row_val,col_val):
    Imgs_trn, Msks_trn = get_data_montage(imgs_trn, msks_trn,nb_rows=row_trn, nb_cols=col_trn, rng=np.random)
    Imgs_val, Msks_val = get_data_montage(imgs_val, msks_val,nb_rows=row_val, nb_cols=col_val, rng=np.random)
    return Imgs_trn,Msks_trn,Imgs_val,Msks_val
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
def random_transforms(items, nb_min=0, nb_max=5, rng=np.random):
    all_transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
        # lambda x: add_noise(x),
    ]
    n = rng.randint(nb_min, nb_max + 1)
    items_t = [item.copy() for item in items]
    for _ in range(n):
        idx = rng.randint(0, len(all_transforms))
        transform = all_transforms[idx]
        items_t = [transform(item) for item in items_t]
    return items_t
def add_noise(img):
    for i in range(200):
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
#########################################################################################
def normalization(imgs):
    #Normalization
    _mean, _std = np.mean(imgs), np.std(imgs)
    normalize = lambda x: (x - _mean) / (_std + 1e-10)
    return normalize(imgs)
def tiff_img_reader(imgs_path,msks_path = None):
    ##tiff img reader
    imgs = tiff.imread(imgs_path).astype('float32') /255
    if msks_path != None:
        msks =tiff.imread(msks_path) // 255
    else:
        msks = None
    return imgs,msks
def data_split(imgs_path,msks_path):
    imgs, msks =  tiff_img_reader(imgs_path,msks_path)
    idxs = random.sample(range(imgs.shape[0]),imgs.shape[0] // 5)
    rest = [i for i in range(imgs.shape[0]) if i not in idxs]
    imgs_val = imgs[idxs]
    msks_val = msks[idxs]
    imgs_trn = imgs[rest]
    msks_trn = msks[rest]
    return imgs_trn,msks_trn,imgs_val,msks_val
def Augment(imgs, msks, input_shape, aug_ration=10, transform = False, rng=np.random):
    #### imgs: dtype float32  msks: dtype uint8 ####
    img_len, H, W = imgs.shape
    wdw_H,wdw_W = input_shape
    img_batch = np.zeros((img_len*aug_ration,) + input_shape, dtype=np.float32)
    msk_batch = np.zeros((img_len*aug_ration,) + input_shape, dtype=np.uint8)
    batch_idx = 0
    for img_idx in range(img_len):
        for num in range(aug_ration):
            y0, x0 = rng.randint(0, H - wdw_H), rng.randint(0, W - wdw_W)
            y1, x1 = y0 + wdw_H, x0 + wdw_W
            im = imgs[img_idx][y0:y1, x0:x1]
            im_mask = msks[img_idx][y0:y1, x0:x1]
            #### elastic  transform
            if np.random.randint(0, 10) > 7:
                im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,im_merge.shape[1] * 0.08)
                im_t = im_merge_t[..., 0]
                im_mask_t = im_merge_t[..., 1]
                img_batch[batch_idx] = im_t
                msk_batch[batch_idx] = im_mask_t
            else:
                img_batch[batch_idx] = im
                msk_batch[batch_idx] = im_mask
            if transform:
                [img_batch[batch_idx], msk_batch[batch_idx]] = random_transforms(
                    [img_batch[batch_idx], msk_batch[batch_idx]])
            batch_idx += 1
    img_batch = img_batch[:, :, :, np.newaxis]
    msk_batch = msk_batch[:, :, :, np.newaxis]
    # tiff.imsave('test.tif',np.uint8(img_batch[0]*255))
    return img_batch,msk_batch
def save_npy(data, name, npy_path):
    print "saving npy data..."
    np.save(npy_path + '/'+name, data)
    print "Data %s saved in root: %s." % (name, npy_path)

def main():
    ### Training Data
    imgs_trn, msks_trn, imgs_val, msks_val = data_split('data/train-volume.tif','data/train-labels.tif')
    img_batch, msk_batch = Augment(imgs_trn, msks_trn,(128,128),100,transform=True)

    # save_npy(img_batch,"img_trn.npy","data")
    # save_npy(msk_batch, "msk_trn.npy", "data")
    # imgs_val = imgs_val[:, :, :, np.newaxis]
    # save_npy(imgs_val, "img_val.npy", "data")
    # msks_val = msks_val[:, :, :, np.newaxis]
    # save_npy(msks_val, "msk_val.npy", "data")

    ### Testing Data
    # imgs_tst,_ = tiff_img_reader('data/test-volume.tif')
    # imgs_tst = imgs_tst[:,:,:,np.newaxis]
    # save_npy(imgs_tst, "img_tst.npy", "data")

if __name__ == "__main__":
    main()





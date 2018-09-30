from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import datetime
from os import path, makedirs
import matplotlib.pyplot as plt
from data_reader import DataLoader
import numpy as np
from network import build_discriminator
from scipy.misc import imsave
from network import  get_generator, get_densenet
from keras.callbacks import TensorBoard
import tensorflow as tf
from utils import dice_coef_loss,dice_coef

def write_log(callback,name,value,batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary,batch_no)
    callback.writer.flush()

class ADDN():
    def __init__(self,checkpoint_name):
        ### Configurations
        self.config = {
            'data_path': "data",
            'input_shape': (128, 128, 1),
            'output_shape': (128, 128, 1),
            'batch_size': 2,
            'epochs': 100,
            'sample_interval': 200,
            'df':64,
            'patch':32
        }
        # Calculate output shape of D (PatchGAN)
        self.img_rows = int(self.config['input_shape'][0])
        self.disc_patch = (self.config['patch'], self.config['patch'], 1)
        self.data_loader = DataLoader(dataset=self.config['data_path'])
        self.checkpoint_name = checkpoint_name

        self.generator = None
        self.discriminator = None
        self.combined = None
        self.imgs_trn = None
        self.msks_trn = None
        self.imgs_val = None
        self.msks_val = None
        log_path = 'Graph/addn'
        self.callback = TensorBoard(log_path)
        return

    @property
    def checkpoint_path(self):
        return 'checkpoints/%s' % (self.checkpoint_name)

    def compile(self):
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = build_discriminator(self.config['input_shape'], self.config['df'])
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # Build the generator
        # self.generator = get_densenet(self.config['input_shape'])
        self.generator =get_generator(self.config['input_shape'])
        img = Input(shape=self.config['input_shape'])
        label = Input(shape=self.config['input_shape'])
        seg = self.generator(img)
        self.discriminator.trainable = False
        valid = self.discriminator([seg, img])
        self.combined = Model(inputs=[label, img], outputs=[valid, seg])
        self.combined.compile(loss=['mse',dice_coef_loss], loss_weights=[1, 100], optimizer=optimizer)
        self.callback.set_model(self.generator)
        return

    def train(self, sample=False):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((self.config['batch_size'],) + self.disc_patch)
        fake = np.zeros((self.config['batch_size'],) + self.disc_patch)
        for epoch in range(self.config['epochs']):
            for batch_i, (imgs, labels) in enumerate(self.data_loader.load_batch(self.config['batch_size'])):
                # Condition on B and generate a translated version

                # Train the discriminators (original images = real / generated = Fake)
                segs = self.generator.predict(imgs)
                d_loss_real = self.discriminator.train_on_batch([labels, imgs], valid)
                d_loss_fake = self.discriminator.train_on_batch([segs, imgs], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # Train the generators
                g_loss = self.combined.train_on_batch([labels, imgs], [valid, labels])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, self.config['epochs'],
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                self.generator.save_weights('%s/weights_loss_trn.weights'% self.checkpoint_path)

                # If at save interval => save generated image samples
                # if sample == True:
                #     if batch_i % self.config['sample_interval'] == 0:
                #         self.sample_images(epoch, batch_i)
            train_names = 'train_loss'
            write_log(self.callback, train_names, g_loss[4], epoch)

            imgs, labels = self.data_loader.load_img(2)
            g_val = self.combined.test_on_batch([labels, imgs], [valid, labels])
            val_names = 'val_acc'
            write_log(self.callback, val_names, g_val[4], epoch)
        return
    def predict(self, imgs):
        return self.generator.predict(imgs)

    def sample_images(self, epoch, batch_i):
        # r, c = 3, 3
        imgs, labels = self.data_loader.load_img(batch_size=1)
        segs = self.predict(imgs)
        print (segs.shape)
        imsave("checkpoints/ADDN/images/%d_%d.tif" % (epoch, batch_i), segs[0][:, :, 0])

def main():
    prs = argparse.ArgumentParser()
    prs.add_argument('--name', help='name used for checkpoints', default='ADDN', type=str)
    # choose the mode
    subprs = prs.add_subparsers(title='actions', description='Choose from one of the actions.')
    subprs_trn = subprs.add_parser('train', help='Run training.')
    subprs_trn.set_defaults(which='train')
    subprs_trn.add_argument('-w', '--weights', help='path to keras weights')

    subprs_sbt = subprs.add_parser('submit', help='Make submission.')
    subprs_sbt.set_defaults(which='submit')
    subprs_sbt.add_argument('-w', '--weights', help='path to keras weights')

    subprs_val = subprs.add_parser('val', help='Make validation.')
    subprs_val.set_defaults(which='val')
    subprs_val.add_argument('-w', '--weights', help='path to keras weights')
    args = vars(prs.parse_args())

    assert args['which'] in ['train', 'submit','val']
    model = ADDN(args['name'])
    if not path.exists(model.checkpoint_path):
        makedirs(model.checkpoint_path)
    def load_weights():
        if args['weights'] is not None:
            model.generator.load_weights(args['weights'])
    if args['which'] == 'train':
        model.compile()
        load_weights()
        model.generator.summary()
        model.train(sample=True)
    elif args['which'] == 'val':
        model.config['input_shape'] = (512, 512, 1)
        model.config['output_shape'] = (512, 512, 1)
        model.compile()
        load_weights()
        imgs_val, _ = model.data_loader.load_npy_data("val")
        msks_sbt = model.predict(imgs_val)
        save_path = os.path.join(model.checkpoint_path, "val")
        if not path.exists(save_path):
            makedirs(save_path)
        for i in range(msks_sbt.shape[0]):
            imsave("%s/%d.tif" % (save_path, i), msks_sbt[i][:, :, 0])
    elif args['which'] == 'submit':
        model.config['input_shape'] = (512, 512, 1)
        model.config['output_shape'] = (512, 512, 1)
        model.compile()
        load_weights()
        imgs_tst = model.data_loader.load_npy_data("submit")
        msks_sbt = model.predict(imgs_tst)
        save_path = os.path.join(model.checkpoint_path, "test")
        if not path.exists(save_path):
            makedirs(save_path)
        for i in range(msks_sbt.shape[0]):
            imsave("%s/%d.tif" % (save_path, i), msks_sbt[i][:, :, 0])
if __name__ == '__main__':
    main()

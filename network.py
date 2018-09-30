from keras.models import Model
from keras.layers import UpSampling2D,Activation,BatchNormalization,Input,Conv2D, MaxPooling2D, Dropout, \
    concatenate, Conv2DTranspose, Concatenate,LeakyReLU
import keras.backend as K

def get_densenet(input_shape):
    K.set_image_dim_ordering('tf')
    x = inputs = Input(shape=input_shape, dtype='float32')
    x = Conv2D(48, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.2)(x)
    x = dc_0_out = dense_block(x)
    x = transition_Down(x, 96)
    x = dc_1_out = dense_block(x)
    x = transition_Down(x, 144)
    x = dc_2_out = dense_block(x)
    x = transition_Down(x, 192)
    x = dc_3_out = dense_block(x)
    x = transition_Down(x, 240)
    x = dense_block(x)
    x = transition_Up(x, 48)
    x = concatenate([x, dc_3_out])
    x = dense_block(x)
    x = transition_Up(x, 48)
    x = concatenate([x, dc_2_out])
    x = dense_block(x)
    x = transition_Up(x, 48)
    x = concatenate([x, dc_1_out])
    x = dense_block(x)
    x = transition_Up(x, 48)
    x = concatenate([x, dc_0_out])
    x = dense_block(x)
    x = Conv2D(1, 1, activation='sigmoid')(x)
    net = Model(inputs=inputs, outputs=x)
    return net

def dense_block(input_layer, features=12, depth=4,temperature=1.0, padding='same', batchnorm=False,dropout=0.2):
    inputs = x = input_layer
    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3, 3)
    for n in range(depth):
        x0 = x
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding, kernel_initializer='he_normal')(x)
        x = Dropout(dropout)(x)
        maps.append(x)
        if n!= depth-1:
            x = Concatenate()([x0, x])
        else:
            x = Concatenate()(maps)
        dilation_rate *= 2
    return x

def get_generator(input_shape):
    K.set_image_dim_ordering('tf')
    x = inputs = Input(input_shape, dtype='float32')
    x = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = dc_0_out = Dropout(0.2)(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = dc_1_out = Dropout(0.2)(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = dc_2_out = Dropout(0.2)(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = dc_3_out = Dropout(0.2)(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(256, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
    x = concatenate([x, dc_3_out])
    x = Dropout(0.2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(128, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
    x = concatenate([x, dc_2_out])
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(64, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
    x = concatenate([x, dc_1_out])
    x = Dropout(0.2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(32, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
    x = concatenate([x, dc_0_out])
    x = Dropout(0.2)(x)
    x = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(1, 1, activation='sigmoid')(x)
    # x = Lambda(lambda x: x[:, :, :, 1], output_shape=output_shape)(x)
    net = Model(inputs=inputs, outputs=x)
    return net

def transition_Down(input_layer,features,kernel_size=(3,3), padding='same',dropout=0.2):
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(features, kernel_size,
                   padding=padding, kernel_initializer='he_normal')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D(2,2)(x)
    return x

def transition_Up(input_layer, feature, kernel_size=(3,3),stride=2):
    x = Conv2DTranspose(feature, 2, strides=stride, activation='relu', kernel_initializer='he_normal')(input_layer)
    return x

def build_discriminator(img_shape,df):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    def layer (layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])
    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = layer(d2, df*4)
    # d4 = layer(d3, df*8)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d3)
    model = Model([img_A, img_B], validity)
    return model
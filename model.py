from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2
import poisson

from ops import *
from utils import *


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
               batch_size=64, batch_complete_size = 64, sample_num=64, output_height=64, output_width=64,
               y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
               gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='celebA',
               input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data', lam=0.1):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.batch_complete_size = batch_complete_size

    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.image_size = output_width
    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.lam = lam

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
      self.data = glob(data_path)
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")
      np.random.shuffle(self.data)
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

      if len(self.data) < self.batch_size:
        raise Exception("[!] Entire dataset size is less than the configured batch_size")

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):

    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      self.image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      self.image_dims = [self.input_height, self.input_width, self.c_dim]

    # self.is_training = tf.placeholder(tf.bool, name='is_training')

    self.inputs = tf.placeholder(
        tf.float32, [self.batch_size] + self.image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
        tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G = self.generator(self.z, self.y)
    self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
    self.sampler = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

    # Completion.
    self.mask = tf.placeholder(tf.float32, self.image_dims, name='mask')
    self.images = tf.placeholder(
        tf.float32, [self.batch_complete_size] + self.image_dims, name='real_images')

    self.contextual_loss = tf.reduce_sum(
        tf.contrib.layers.flatten(
            tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)

    self.comp = tf.multiply(1.0-self.mask, self.images) + tf.multiply(self.mask, self.G)

    self.D_complete, self.D_logits_complete = self.discriminator(self.comp, self.y, reuse=True)
    self.g_loss_complete = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_complete, tf.ones_like(self.D_complete)))

    self.perceptual_loss = self.g_loss_complete + 0.8*self.g_loss
    self.complete_loss = self.contextual_loss + self.lam * self.g_loss + 2* self.lam * self.g_loss_complete
    self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:
        self.data = glob(os.path.join(
            config.data_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(self.data)
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, int(batch_idxs)):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
        else:
          batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
            .astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
                                         feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y: batch_labels,
          })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={
              self.z: batch_z,
              self.y: batch_labels,
          })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={self.z: batch_z, self.y: batch_labels})
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
                                         feed_dict={self.inputs: batch_images, self.z: batch_z})
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={self.z: batch_z})
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={self.z: batch_z})
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({self.z: batch_z})
          errD_real = self.d_loss_real.eval({self.inputs: batch_images})
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
              % (epoch, config.epoch, idx, batch_idxs,
                 time.time() - start_time, errD_fake + errD_real, errG))

        if np.mod(counter, 100) == 1:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                    self.y: sample_labels,
                }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                  [self.sampler, self.d_loss, self.g_loss],
                  feed_dict={
                      self.z: sample_z,
                      self.inputs: sample_inputs,
                  },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                          './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            except:
              print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def complete(self, config):

    # Create new directries
    #def make_dir(name):
          # Works on python 2.7, where exist_ok arg to makedirs isn't available.
    #      p = os.path.join(config.outDir, name)
    #      if not os.path.exists(p):
    #          os.makedirs(p)
    #  make_dir('hats_imgs')
    #  make_dir('completed')
     # make_dir('logs')

    # Initialization

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    # Load the pre-trained DCGAN
    isLoaded = self.load(self.checkpoint_dir)
    assert(isLoaded)

    # Number of images to be completed
    nImgs = config.num


    # Index of batches to be completed
    batch_idxs = int(np.ceil(nImgs/self.batch_complete_size))


    # Generate different masks
    # Random
    if config.maskType == 'random':
        fraction_masked = 0.8
        mask = np.ones(self.image_dims)
        mask[np.random.random(self.image_dims[:2]) < fraction_masked] = 0.0
        center = None
    # Center
    elif config.maskType == 'center':
        assert(config.centerScale <= 0.5)
        mask = np.ones(self.image_dims)
        sz = self.image_size
        l = int(self.image_size*config.centerScale)
        u = int(self.image_size*(1.0-config.centerScale))
        mask[l:u, l:u, :] = 0.0
        center = (int((l+u)/2),int((l+u)/2))
    # Eye
    elif config.maskType == 'eye':
        assert(config.centerScale <= 0.5)
        mask = np.ones(self.image_dims)
        sz = self.image_size
        l = int(self.image_size*config.centerScale)
        u = int(self.image_size*(1.0-config.centerScale))
        mask[l:u, :, :] = 0.0
        center = (int((l+u)/2),int((l+u)/2))
    # Left half
    elif config.maskType == 'left':
        mask = np.ones(self.image_dims)
        c = self.image_size // 2
        mask[:,:c,:] = 0.0
        center = (int(c/2),c)
    # No mask
    elif config.maskType == 'full':
        mask = np.ones(self.image_dims)
        center = None
    # Grid mask
    elif config.maskType == 'crop':
        assert(config.cropScale >= 0.5)
        mask = np.zeros(self.image_dims)
        sz = self.image_size
        l = int(self.image_size*(1.0-config.cropScale))
        u = int(self.image_size*(config.cropScale))
        mask[l:u, l:u, :] = 1.0
        center = (int((l+u)/2),int((l+u)/2))
    else:
        assert(False)

    #all_images = glob(config.imgs[0])

    for idx in xrange(0, batch_idxs):
        # Load images for this batch
        if self.dataset_name == 'celebA':
          all_images = glob(config.imgs[0])
          l = idx*self.batch_complete_size
          u = min((idx+1)*self.batch_complete_size, nImgs)
          batchSz = u-l
          batch_files = all_images[l:u]
          batch = [get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale)
                  for batch_file in batch_files]

          batch_images = np.array(batch).astype(np.float32)
          output = np.zeros(batch_images.shape)
        else:
          all_images, labels = self.load_mnist()
          l = idx*self.batch_complete_size
          u = min((idx+1)*self.batch_complete_size, nImgs)
          batchSz = u-l
          batch_images = all_images[l:u]
          batch_labels = labels[l:u]
          output = np.zeros(batch_images.shape)

        # Pad images if there is no enough images for this batch
        if batchSz < self.batch_complete_size:
            print(batchSz)
            padSz = ((0, int(self.batch_complete_size-batchSz)), (0,0), (0,0), (0,0))
            batch_images = np.pad(batch_images, padSz, 'constant')
            batch_images = batch_images.astype(np.float32)


        # Sample the z input vector
        zhats = np.random.uniform(-1, 1, size=(self.batch_complete_size, self.z_dim))
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1
        m = 0
        v = 0


        nRows = int(np.ceil(batchSz**(.5)))
        nCols = int(min(batchSz**(.5), batchSz))
        save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                    os.path.join(config.outDir, 'before.png'))

        if self.dataset_name == 'celebA':
          masked_images = np.multiply(inverse_transform(batch_images), mask)
          save_images(masked_images[:batchSz,:,:,:]*2-1, [nRows,nCols],
                    os.path.join(config.outDir, 'masked.png'))
        else:
          masked_images = np.multiply(batch_images, mask)
          save_images(masked_images[:batchSz,:,:,:]*2-1, [nRows,nCols],
                    os.path.join(config.outDir, 'masked.png'))

        # If low resolution
#        if lowres_mask.any():
#            lowres_images = np.reshape(batch_images, [self.batch_size, self.lowres_size, self.lowres,
#                self.lowres_size, self.lowres, self.c_dim]).mean(4).mean(2)
#            lowres_images = np.multiply(lowres_images, lowres_mask)
#            lowres_images = np.repeat(np.repeat(lowres_images, self.lowres, 1), self.lowres, 2)
#            save_images(lowres_images[:batchSz,:,:,:], [nRows,nCols], os.path.join(config.outDir, 'lowres.png'))

        for img in range(batchSz):
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                    f.write('iter loss ' +
                            ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) +
                            '\n')


        for i in xrange(config.nIter):
            if self.dataset_name == 'celebA':
              fd = {
                  self.z: zhats,
                  self.mask: mask,
                  self.images: batch_images,
              }
              run = [self.complete_loss, self.grad_complete_loss, self.G]
              loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
            else:
              fd = {
                  self.z: zhats,
                  self.mask: mask,
                  self.y: batch_labels,
                  self.images: batch_images,
              }
              run = [self.complete_loss, self.grad_complete_loss, self.G]
              loss, g, G_imgs = self.sess.run(run, feed_dict=fd)


            for img in range(batchSz):
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                   f.write('{} {} '.format(i, loss[img]).encode())
                   np.savetxt(f, zhats[img:img+1])


            print(i)
            if i % config.outInterval == 0:
                print(i, np.mean(loss[0:batchSz]))
                imgName = os.path.join(config.outDir,
                                        'hats_imgs/{:04d}.png'.format(i))
                nRows = int(np.ceil(batchSz**(.5)))
                nCols = int(min(batchSz**(.5), batchSz))
                save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                if self.dataset_name == 'celebA':
                  inv_masked_hat_images = np.multiply((G_imgs+1)/2, 1.0-mask)
                  completed = masked_images + inv_masked_hat_images
                else:
                  inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                  completed = masked_images + inv_masked_hat_images

                completed = completed*2-1

                for j in range(batchSz):
                    if self.dataset_name == 'celebA':
                      channels = 3
                    else:
                      channels = 1

                    # Call the poisson method on each individual channel
                    if self.dataset_name == 'celebA':
                      result_stack = [poisson.process(127.5*(G_imgs[j,:,:,k]+1), 127.5*(completed[j,:,:,k]+1), 1-mask[:,:,0]) for k in range(channels)]
                    else:
                      result_stack = [poisson.process(255*(G_imgs[j,:,:,k]), 255*(completed[j,:,:,k]), 1-mask[:,:,0]) for k in range(channels)]

                    if self.dataset_name == 'celebA':
                      output[j,:,:,:] = cv2.merge(result_stack)
                    else:
                      output[j,:,:,0] = cv2.merge(result_stack)

                output[output<0] = 0
                output[output>255] = 255
                output = output/127.5 - 1

                imgName = os.path.join(config.outDir,
                                        'completed/{:04d}.png'.format(i))
                save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

                if self.dataset_name == 'celebA':
                  imgName = os.path.join(config.outDir,
                                        'completed_blend/{:04d}.png'.format(i))
                  save_images(output[:batchSz,:,:,:], [nRows,nCols], imgName)


            m_y = 0
            v_y = 0
            if config.approach == 'adam':
                # Optimize single completion with Adam
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - config.beta1 ** (i + 1))
                v_hat = v / (1 - config.beta2 ** (i + 1))
                zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                zhats = np.clip(zhats, -1, 1)

            elif config.approach == 'hmc':
                # Sample example completions with HMC (not in paper)
                zhats_old = np.copy(zhats)
                loss_old = np.copy(loss)
                v = np.random.randn(self.batch_complete_size, self.z_dim)
                v_old = np.copy(v)

                for steps in range(config.hmcL):
                    v -= config.hmcEps/2 * config.hmcBeta * g[0]
                    zhats += config.hmcEps * v
                    np.copyto(zhats, np.clip(zhats, -1, 1))
                    loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                    v -= config.hmcEps/2 * config.hmcBeta * g[0]

                for img in range(batchSz):
                    logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                    logprob = config.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                    accept = np.exp(logprob_old - logprob)
                    if accept < 1 and np.random.uniform() > accept:
                        np.copyto(zhats[img], zhats_old[img])

                config.hmcBeta *= config.hmcAnneal
            else:
                assert(False)

  def present(self, config):

    # Initialization
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    # Load the pre-trained DCGAN
    isLoaded = self.load(self.checkpoint_dir)
    assert(isLoaded)

    # Number of images to be completed
    nImgs = config.num


    # Index of batches to be completed
    mask_idxs = 6

    # Generate different masks
    mask = np.ones([mask_idxs, self.output_height, self.output_width, self.c_dim])
    output = np.zeros(mask.shape)

    # Random
    fraction_masked = 0.8
    mask[0, np.random.random(self.image_dims[:2]) < fraction_masked] = 0.0

    # Center
    sz = self.image_size
    l = int(self.image_size*config.centerScale)
    u = int(self.image_size*(1.0-config.centerScale))
    mask[2, l:u, l:u, :] = 0.0

    # Eye
    sz = self.image_size
    l = int(self.image_size*config.centerScale)
    u = int(self.image_size*(1.0-config.centerScale))
    mask[3, l:u, :, :] = 0.0

    # Left half
    c = self.image_size // 2
    mask[4,:,:c,:] = 0.0

    # Grid mask
    mask[1,:,:,:] = 0
    mask[1,::2,::2,:] = 1.0

    # Crop
    sz = self.image_size
    l = int(self.image_size*(1.0-config.cropScale))
    u = int(self.image_size*(config.cropScale))
    mask[5,:,:,:] = 0
    mask[5,l:u, l:u, :] = 1.0


    all_images = glob(config.imgs[0])

    batch_files = all_images[0:4]

    batch = [get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale)
                  for batch_file in batch_files]


    batch_images = np.array(batch).astype(np.float32)
    batch_images = batch_images[3]

    # Sample the z input vector
    zhats = np.random.uniform(-1, 1, size=(mask_idxs, self.z_dim))
    m = np.zeros((mask_idxs,self.z_dim))
    v = np.zeros((mask_idxs,self.z_dim))
    nRows = 3
    nCols = 2
    G_imgs = np.zeros([mask_idxs, self.output_height, self.output_width, self.c_dim])
    #save_images([batch_images], [1,1],
    #            os.path.join(config.outDir, 'before.png'))


    masked_images = np.multiply(mask, inverse_transform(batch_images))
    save_images(masked_images[:mask_idxs,:,:,:]*2-1, [nRows,nCols],
                os.path.join(config.outDir, 'masked.png'))


    for i in xrange(config.nIter):

        for k in range(mask_idxs):

          fd = {
              self.z: np.expand_dims(zhats[k,:],axis=0),
              self.mask: mask[k,:,:,:],
              self.images: np.expand_dims(batch_images,axis=0),
          }
          run = [self.complete_loss, self.grad_complete_loss, self.G]
          loss, g, G  = self.sess.run(run, feed_dict=fd)

          G_imgs[k,:,:,:] = G[0,:,:,:]

          print(i, k, np.mean(loss))

          # Optimize single completion with Adam
          m_prev = np.copy(m[k])
          v_prev = np.copy(v[k])

          m[k] = config.beta1 * m_prev + (1 - config.beta1) * g[0]
          v[k] = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
          m_hat = m[k] / (1 - config.beta1 ** (i + 1))
          v_hat = v[k] / (1 - config.beta2 ** (i + 1))
          zhats[k,:] += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
          zhats[k,:] = np.clip(zhats[k,:], -1, 1)

        if i % config.outInterval == 0:
            imgName = os.path.join(config.outDir,
                                        'hats_imgs/{:04d}.png'.format(i))
            nRows = 3
            nCols = 2
            save_images(G_imgs[:mask_idxs,:,:,:], [nRows,nCols], imgName)

            G_imgs = (G_imgs+1)/2
            inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
            completed = masked_images + inv_masked_hat_images
            completed = completed *2 - 1

            for j in range(mask_idxs):
                channels = 3
                # Call the poisson method on each individual channel
                result_stack = [poisson.process(255*G_imgs[j,:,:,kk], 255*(masked_images[j,:,:,kk]), 1-mask[j,:,:,0]) for kk in range(channels)]
                output[j,:,:,:] = cv2.merge(result_stack)


            output[output<0] = 0
            output[output>255] = 255

            imgName = os.path.join(config.outDir,
                                        'completed/{:04d}.png'.format(i))
            save_images(completed[:mask_idxs,:,:,:], [nRows,nCols], imgName)

            imgName = os.path.join(config.outDir,
                                        'completed_blend/{:04d}.png'.format(i))
            save_images(output[:mask_idxs,:,:,:], [nRows,nCols], imgName)




  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])
        h1 = concat([h1, y], 1)

        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
        s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
        s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.dataset_name)


    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)


      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

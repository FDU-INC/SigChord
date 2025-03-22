# Copyright 2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.io import savemat
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.keras.backend.clear_session()

import models.benchmark.dcs.dcs_tf.cs as cs
import models.benchmark.dcs.dcs_tf.file_utils as file_utils
import models.benchmark.dcs.dcs_tf.utils as utils
import models.benchmark.dcs.dcs_tf.get_data as get_data

#tf.enable_eager_execution()

tfd = tfp.distributions

flags.DEFINE_string(
    'mode', 'recons', 'Model mode.')
flags.DEFINE_integer(
    'num_fitting_iterations', 20000,
    'Number of fitting iterations.')
flags.DEFINE_integer(
    'batch_size', 8, 'Fitting batch size.')
flags.DEFINE_integer(
    'num_measurements', 32, 'The number of measurements')
flags.DEFINE_integer(
    'num_latents', 25, 'The number of latents')
flags.DEFINE_integer(
    'num_z_iters', 3, 'The number of latent optimisation steps.')
flags.DEFINE_float(
    'z_step_size', 0.01, 'Step size for latent optimisation.')
flags.DEFINE_string(
    'z_project_method', 'norm', 'The method to project z.')
flags.DEFINE_integer(
    'summary_every_step', 1000,
    'The interval at which to log debug ops.')
flags.DEFINE_integer(
    'export_every', 6000,
    'The interval at which to export samples.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_string(
    'output_dir', './output', 'Location where to save output files.')

flags.DEFINE_integer('item_index',5,'The item index.')
flags.DEFINE_integer('nChannels',8,'The number of channels.')
flags.DEFINE_integer('use_noise',10,'Noise snr.')

FLAGS = flags.FLAGS

# Log info level (for Hooks).
tf.logging.set_verbosity(tf.logging.INFO)

n_avail_bands = 16
n_total_bands = 40

def main(argv):
  del argv

  utils.make_output_dir(FLAGS.output_dir)
  data_processor = utils.DataProcessor()
  signals,scale = utils.get_single_signal_dataset(
      FLAGS.batch_size,FLAGS.item_index,FLAGS.use_noise)

  logging.info('Learning rate: %d', FLAGS.learning_rate)

  # Construct optimizers.
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  # Create the networks and models.
  generator = utils.get_generator()
  model = cs.CS(generator,
                FLAGS.num_z_iters, FLAGS.z_step_size, FLAGS.z_project_method)
  prior = utils.make_prior(FLAGS.num_latents)
  generator_inputs = prior.sample(FLAGS.batch_size)

  A = get_data.getA(FLAGS.nChannels)

  model_output = model.connect(signals, generator_inputs,A)
  optimization_components = model_output.optimization_components
  debug_ops = model_output.debug_ops

  signals = utils.complex_to_real(signals)
  reconstructions,_ = utils.optimise_and_sample(
      generator_inputs, model, signals,False,A)
  signals = utils.real_to_complex(signals)
  reconstructions = utils.real_to_complex(reconstructions)

  global_step = tf.train.get_or_create_global_step()
  update_op = optimizer.minimize(
      optimization_components.loss,
      var_list=optimization_components.vars,
      global_step=global_step)

  sample_exporter = file_utils.FileExporter(
      os.path.join(FLAGS.output_dir, 'reconstructions'))

  # Hooks.
  debug_ops['it'] = global_step
  # Abort training on Nans.
  nan_hook = tf.train.NanTensorHook(optimization_components.loss)
  # Step counter.
  step_conter_hook = tf.train.StepCounterHook()

  checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=utils.get_ckpt_dir(FLAGS.output_dir), save_secs=10 * 60)

  loss_summary_saver_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.summary_every_step,
      output_dir=os.path.join(FLAGS.output_dir, 'summaries'),
      summary_op=utils.get_summaries(debug_ops))

  hooks = [step_conter_hook,loss_summary_saver_hook]

  # Start fitting.

  with tf.train.MonitoredSession(hooks = hooks) as sess:
    logging.info('starting fitting')

    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    mse_ls = []
    filepath = (f'{FLAGS.nChannels}_n_{FLAGS.use_noise}.txt')
    start_time = time.time()
    last_time = start_time
    for i in range(FLAGS.num_fitting_iterations):
      _,signals_np,reconstructions_np=sess.run([update_op,signals,reconstructions])
      bias = i % (2400//FLAGS.batch_size) #[0-299]
      if bias == 0:
        gt_signals = np.zeros((2400, n_avail_bands), dtype=np.complex64)
        recon_signals = np.zeros((2400, n_avail_bands), dtype=np.complex64)

      gt_signals[bias*FLAGS.batch_size:(bias+1)*FLAGS.batch_size,:] = signals_np
      recon_signals[bias*FLAGS.batch_size:(bias+1)*FLAGS.batch_size,:] = reconstructions_np


      if bias == 2400/FLAGS.batch_size-1:
        r = recon_signals
        r = r.T
        r = r.reshape([-1]) * scale * (n_total_bands*2400)
        r = np.pad(r, (0, (n_total_bands - n_avail_bands)*2400), mode="constant")
        r_temp = np.fft.ifft(r)

        s = gt_signals
        s = s.T
        s = s.reshape([-1]) * scale * (n_total_bands * 2400)
        s = np.pad(s, (0, (n_total_bands - n_avail_bands)*2400), mode="constant")
        s_temp = np.fft.ifft(s)

        if (i+1)%FLAGS.export_every ==0:
          plt.subplot(2, 2, 1)
          plt.plot(s_temp.real)
          plt.plot(s_temp.imag)
          plt.subplot(2, 2, 2)
          plt.plot(r_temp.real)
          plt.plot(r_temp.imag)
          plt.subplot(2, 2, 3)
          plt.plot(np.abs(s))
          plt.subplot(2, 2, 4)
          plt.plot(np.abs(r))
          plt.savefig(f"./recons/rimage_epoch_{i + 1}.png")
          plt.clf()

        s_temp /= np.sqrt(np.mean(np.abs(s_temp) ** 2))
        r_temp /= np.sqrt(np.mean(np.abs(r_temp) ** 2))
        mse = np.mean(np.abs(s_temp - r_temp) ** 2)
        print("mse:",mse)
        mse_ls.append(mse)
        this_time = time.time()
        print("time:", this_time-last_time)
        last_time = this_time

    mse_min = min(mse_ls)
    print("total_time:", time.time() - start_time)

if __name__ == '__main__':
    app.run(main)
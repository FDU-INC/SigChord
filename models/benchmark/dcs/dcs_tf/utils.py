
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
"""Tools for latent optimisation."""
import collections
import os

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from models.benchmark.dcs.dcs_tf import nets, get_data

tfd = tfp.distributions


class ModelOutputs(
    collections.namedtuple('AdversarialModelOutputs',
                           ['optimization_components', 'debug_ops'])):
  """All the information produced by the adversarial module.

  Fields:

    * `optimization_components`: A dictionary. Each entry in this dictionary
      corresponds to a module to train using their own optimizer. The keys are
      names of the components, and the values are `common.OptimizationComponent`
      instances. The keys of this dict can be made keys of the configuration
      used by the main train loop, to define the configuration of the
      optimization details for each module.
    * `debug_ops`: A dictionary, from string to a scalar `tf.Tensor`. Quantities
      used for tracking training.
  """


class OptimizationComponent(
    collections.namedtuple('OptimizationComponent', ['loss', 'vars'])):
  """Information needed by the optimizer to train modules.

  Usage:
      `optimizer.minimize(
          opt_compoment.loss, var_list=opt_component.vars)`

  Fields:

    * `loss`: A `tf.Tensor` the loss of the module.
    * `vars`: A list of variables, the ones which will be used to minimize the
      loss.
  """

def measure(X,A):
  X = tf.cast(X, dtype=tf.complex64)
  A = tf.cast(A, dtype=tf.complex64)
  # 计算 X 与 W^T 的乘积
  Y = tf.matmul(X, A, transpose_b=True)

  return Y


def complex_to_real(input_tensor):
  # 提取实部和虚部
  real_part = tf.real(input_tensor)
  imag_part = tf.imag(input_tensor)

  # 将实部和虚部交替拼接
  real_imag_tensor = tf.stack([real_part, imag_part], axis=-1)  # 将实部和虚部堆叠在一起

  # 通过reshape将堆叠后的结果交替排布
  real_imag_tensor = tf.reshape(real_imag_tensor, [-1, 2*input_tensor.shape[-1]])
  return real_imag_tensor

def real_to_complex(input_tensor):
  return tf.complex(input_tensor[:,::2], input_tensor[:,1::2])

def gen_loss_fn(data,samples,A):
  data = real_to_complex(data)
  samples = real_to_complex(samples)
  m_data = measure(data,A)
  m_samples = measure(samples,A)
  m_data = complex_to_real(m_data)
  m_samples = complex_to_real(m_samples)
  return tf.reduce_mean(tf.reduce_sum(tf.square(m_data - m_samples), -1))


def cross_entropy_loss(logits, expected):
  """The cross entropy classification loss between logits and expected values.

  The loss proposed by the original GAN paper: https://arxiv.org/abs/1406.2661.

  Args:
    logits: a `tf.Tensor`, the model produced logits.
    expected: a `tf.Tensor`, the expected output.

  Returns:
    A scalar `tf.Tensor`, the average loss obtained on the given inputs.

  Raises:
    ValueError: if the logits do not have shape [batch_size, 2].
  """

  num_logits = logits.get_shape()[1]
  if num_logits != 2:
    raise ValueError(('Invalid number of logits for cross_entropy_loss! '
                      'cross_entropy_loss supports only 2 output logits!'))
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=expected))


def optimise_and_sample(init_z, module, data, is_training,A):
  """Optimising generator latent variables and sample."""
  # print(data.shape)
  if module.num_z_iters is None or module.num_z_iters == 0:
    z_final = init_z
  else:
    init_loop_vars = (0, _project_z(init_z, module.z_project_method))
    loop_cond = lambda i, _: i < module.num_z_iters
    def loop_body(i, z):
      loop_samples = module.generator(z, is_training)
      sparsity_loss = tf.reduce_mean(tf.abs(loop_samples))
      gen_loss = gen_loss_fn(data,loop_samples,A)+sparsity_loss
      #gen_loss = module.gen_loss_fn(data, loop_samples)
      z_grad = tf.gradients(gen_loss, z)[0]
      z -= module.z_step_size * z_grad
      z = _project_z(z, module.z_project_method)
      return i + 1, z

    # Use the following static loop for debugging
    # z = init_z
    # for _ in xrange(num_z_iters):
    #   _, z = loop_body(0, z)
    # z_final = z

    _, z_final = tf.while_loop(loop_cond,
                               loop_body,
                               init_loop_vars)

  return module.generator(z_final, is_training), z_final

def get_optimisation_cost(initial_z, optimised_z):
  optimisation_cost = tf.reduce_mean(
      tf.reduce_sum((optimised_z - initial_z)**2, -1))
  return optimisation_cost


def _project_z(z, project_method='clip'):
  """To be used for projected gradient descent over z."""
  if project_method == 'norm':
    z_p = tf.nn.l2_normalize(z, axis=-1)
  elif project_method == 'clip':
    z_p = tf.clip_by_value(z, -1, 1)
  else:
    raise ValueError('Unknown project_method: {}'.format(project_method))
  return z_p


class DataProcessor(object):

  def preprocess(self, x):
    return x * 2 - 1

  def postprocess(self, x):
    return (x + 1) / 2.


def _get_np_data(data_processor, dataset, split='train'):
  """Get the dataset as numpy arrays."""
  index = 0 if split == 'train' else 1
  if dataset == 'mnist':
    # Construct the dataset.
    x, _ = tf.keras.datasets.mnist.load_data()[index]
    # Note: tf dataset is binary so we convert it to float.
    x = x.astype(np.float32)
    x = x / 255.
    x = x.reshape((-1, 28, 28, 1))

  if dataset == 'cifar':
    x, _ = tf.keras.datasets.cifar10.load_data()[index]
    x = x[:5000]
    x = x.astype(np.float32)
    x = x / 255.

  if data_processor:
    # Normalize data if a processor is given.
    x = data_processor.preprocess(x)
  return x


def make_output_dir(output_dir):
  logging.info('Creating output dir %s', output_dir)
  if not tf.gfile.IsDirectory(output_dir):
    tf.gfile.MakeDirs(output_dir)


def get_ckpt_dir(output_dir):
  ckpt_dir = os.path.join(output_dir, 'ckpt')
  if not tf.gfile.IsDirectory(ckpt_dir):
    tf.gfile.MakeDirs(ckpt_dir)
  return ckpt_dir


def get_real_data_for_eval(num_eval_samples, dataset, split='valid'):
  data = _get_np_data(data_processor=None, dataset=dataset, split=split)
  data = data[:num_eval_samples]
  return tf.constant(data)


def get_summaries(ops):
  summaries = []
  for name, op in ops.items():
    # Ensure to log the value ops before writing them in the summary.
    # We do this instead of a hook to ensure IS/FID are never computed twice.
    print_op = tf.print(name, [op], output_stream=tf.logging.info)
    with tf.control_dependencies([print_op]):
      summary = tf.summary.scalar(name, op)
      summaries.append(summary)

  return summaries

def get_single_signal_dataset(batch_size,item_index,use_noise):
  """Creates the training data tensors."""
  X_train,scale = get_data.get_single_signal(item_index,use_noise)
  train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
  train_dataset = train_dataset.repeat()
  train_iterator = train_dataset.batch(batch_size).make_one_shot_iterator()
  train_data = train_iterator.get_next()
  return train_data,scale


def get_generator():
    return nets.MLPGeneratorNet()


def get_metric_net(dataset, num_outputs=2, use_sn=True):
  if dataset == 'mnist':
    return nets.MLPMetricNet(num_outputs)
  if dataset == 'cifar':
    return nets.ConvMetricNet(num_outputs, use_sn)


def make_prior(num_latents):
  # Zero mean, unit variance prior.
  prior_mean = tf.zeros(shape=(num_latents), dtype=tf.float32)
  prior_scale = tf.ones(shape=(num_latents), dtype=tf.float32)

  return tfd.Normal(loc=prior_mean, scale=prior_scale)

def gaussian(input_tensor):
  # Step 1: reshape [x, 28, 28, 1] -> [x, 784]
  flattened_tensor = tf.reshape(input_tensor, [-1, 784])

  # Step 2: 生成一个 [784, 25] 的随机高斯矩阵
  random_gaussian_matrix = tf.random_normal([784, 100], mean=0.0, stddev=1.0,seed=135)

  # Step 3: 通过矩阵乘法将 [x, 784] 转换为 [x, 25]
  output_tensor = tf.matmul(flattened_tensor, random_gaussian_matrix)

  # 查看最终的形状
  print(output_tensor)

  return output_tensor

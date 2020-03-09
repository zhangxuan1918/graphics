#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module implements the visual hull voxel rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def render(voxels, name=None):
  """Renders the visual hull of a voxel grid.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    voxels: A tensor of shape `[A1, ..., An, Vx, Vy, Vz, Vd]`, where Vx, Vy, Vz
      is the dimension of the voxel grid and Vd the dimension of the
      cells.
    name: A name for this op. Defaults to "visual_hull_render".

  Returns:
    A tensor of shape `[A1, ..., An, Vy, Vz, Vd]`.

  Raises:
    ValueError: If the shape of the input tensors are not supported.
  """
  with tf.compat.v1.name_scope(name, "visual_hull_render", [voxels]):
    voxels = tf.convert_to_tensor(value=voxels)

    shape.check_static(
        tensor=voxels, tensor_name="voxels", has_rank_greater_than=3)

    # Kosta's implementation
    #
    #     def tf_visual_hul(voxels, image_shape=(128, 128, 3)):
    #       image = tf.reduce_sum(voxels, axis=2)
    #       image = tf.ones(image_shape)-tf.exp(-image)
    #       return image
    #
    #
    #     def tf_emission_absorption(voxels, density_factor=100,
    #                                sample_size_z=1024):
    #       signal, density = tf.split(voxels, [3, 1], axis=-1)
    #
    #       density = density * density_factor
    #       density = density / sample_size_z
    #
    #       transmission = tf.cumprod(1.0 - density, axis=-2)
    #
    #       weight = density * transmission
    #       weight_sum = tf.reduce_sum(weight, axis=-2)
    #
    #       rendering = tf.reduce_sum(weight * signal, axis=-2)
    #       rendering = rendering / (weight_sum + 1e-8)
    #
    #       alpha = 1.0 - tf.reduce_prod(1 - density, axis=-2)
    #
    #       return rendering, alpha

    return voxels


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()

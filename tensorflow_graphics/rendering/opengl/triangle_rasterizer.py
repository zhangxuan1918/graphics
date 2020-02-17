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
"""This module implements a differentiable rasterizer of triangular meshes.

The resulting rendering contain perspective-correct interpolation of attributes
defined at the vertices of the rasterized meshes. This rasterizer does not
provide gradients thought visibility, but it does through attributes, allowing
to for instance optimize over geometry, appearance, and in generally speaking
over any 'neural' attribute.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.rendering.opengl import gen_rasterizer_op as render_ops
from tensorflow_graphics.rendering.opengl import math as glm
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape

# TODO(b/149683925): Put the shaders in separate files for reusability &
# code cleanliness.

# Empty vertex shader.
vertex_shader = """
#version 430
void main() { }
"""

# Geometry shader that projects the vertices of visible triangles onto the image
# plane.
geometry_shader = """
#version 430

uniform mat4 view_projection_matrix;

layout(points) in;
layout(triangle_strip, max_vertices=3) out;

out layout(location = 0) vec3 vertex_position;
out layout(location = 1) vec2 barycentric_coordinates;
out layout(location = 2) float triangle_index;

in int gl_PrimitiveIDIn;
layout(binding=0) buffer triangular_mesh { float mesh_buffer[]; };


vec3 get_vertex_position(int vertex_index) {
  int offset = gl_PrimitiveIDIn * 9 + vertex_index * 3;
  return vec3(mesh_buffer[offset], mesh_buffer[offset + 1],
    mesh_buffer[offset + 2]);
}

bool is_back_facing(vec3 vertex_0, vec3 vertex_1, vec3 vertex_2) {
  vec4 tv0 = view_projection_matrix * vec4(vertex_0, 1.0);
  vec4 tv1 = view_projection_matrix * vec4(vertex_1, 1.0);
  vec4 tv2 = view_projection_matrix * vec4(vertex_2, 1.0);
  tv0 /= tv0.w;
  tv1 /= tv1.w;
  tv2 /= tv2.w;
  vec2 a = (tv1.xy - tv0.xy);
  vec2 b = (tv2.xy - tv0.xy);
  return (a.x * b.y - b.x * a.y) <= 0;
}

void main() {
  vec3 vertex_0 = get_vertex_position(0);
  vec3 vertex_1 = get_vertex_position(1);
  vec3 vertex_2 = get_vertex_position(2);

  // Cull back-facing triangles.
  if (is_back_facing(vertex_0, vertex_1, vertex_2)) {
    return;
  }

  vec3 positions[3] = {vertex_0, vertex_1, vertex_2};
  for (int i = 0; i < 3; ++i) {
    // gl_Position is a pre-defined size 4 output variable.
    gl_Position = view_projection_matrix * vec4(positions[i], 1);
    barycentric_coordinates = vec2(i==0 ? 1 : 0, i==1 ? 1 : 0);
    triangle_index = gl_PrimitiveIDIn;

    vertex_position = positions[i];
    EmitVertex();
  }
  EndPrimitive();
}
"""

# Fragment shader that packs barycentric coordinates, triangle index, and depth
# map in a resulting vec4 per pixel.
fragment_shader = """
#version 420

in layout(location = 0) vec3 vertex_position;
in layout(location = 1) vec2 barycentric_coordinates;
in layout(location = 2) float triangle_index;

out vec4 output_color;

void main() {
  output_color = vec4(barycentric_coordinates, round(triangle_index),
    vertex_position.z);
}
"""


class TriangleRasterizer(object):
  """A class allowing to rasterize triangular mesh.

  The resulting images contain perspective-correct interpolation of attributes
  defined at the vertices of the rasterized meshes. Attributes can be defined as
  arbitrary K-dimensional values, which includes depth, appearance,
  'neural features', etc.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Attributes:
    background_geometry: A tensor of shape `[T, 3, 3]` containing `T` triangles,
      each defined by a triplet of 3D vertices, that will serve as background
      for all renderings.
    background_attribute: A tensor of shape `[T, 3, K]` containing batches of
      `T` triangles, with their vertices associated with K-dimensional
      attributes. Pixels for which the first visible surface is in the
      background geometry will make use of `background_attribute` for estimating
      their own attribute.
    camera_origin: A Tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the 3D position of the camera.
    look_at: A Tensor of shape `[A1, ..., An, 3]`, with the last dimension
      storing the position where the camera is looking at.
    camera_up: A Tensor of shape `[A1, ..., An, 3]`, where the last dimension
      defines the up vector of the camera.
    field_of_view:  A Tensor of shape `[A1, ..., An, 1]`, where the last
      dimension represents the vertical field of view of the frustum expressed
      in radians. Note that values for `vertical_field_of_view` must be in the
      range (0,pi).
    image_size_float: A tuple of shape `(W,H)` where `W` and `H` are
      respectively the width and height (in pixels) of the rasterized image.
    image_size_int: An integer version of `image_size_float`.
    near_plane: A Tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the near clipping plane. Note
      that values for `near_plane` must be non-negative.
    far_plane: A Tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the far clipping plane. Note
      that values for `far_plane` must be non-negative.
    bottom_left: A Tensor of shape `[A1, ..., An, 2]`, where the last dimension
      captures the position (in pixels) of the lower left corner of the screen.
    pixel_position: A Tensor of shape `[W, H, 2]` defining the position of all
      the pixels in an image of width `W` and height `H`.
    view_projection_matrix: A Tensor of shape `[A1, ..., An, 4, 4]` transforming
      points from model to homogeneous clip coordinates.
  """

  def __init__(self,
               background_geometry,
               background_attribute,
               camera_origin,
               look_at,
               camera_up,
               field_of_view,
               image_size,
               near_plane,
               far_plane,
               bottom_left,
               name=None):
    """Inits TriangleRasterizer with OpenGL parameters and the background.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      background_geometry: A tensor of shape `[T, 3, 3]` containing `T`
        triangles, each defined by a triplet of 3D vertices, that will serve as
        background for all renderings.
      background_attribute: A tensor of shape `[T, 3, K]` containing batches of
        `T` triangles, with their vertices associated with K-dimensional
        attributes. Pixels for which the first visible surface is in the
        background geometry will make use of `background_attribute` for
        estimating their own attribute.
      camera_origin: A Tensor of shape `[A1, ..., An, 3]`, where the last
        dimension represents the 3D position of the camera.
      look_at: A Tensor of shape `[A1, ..., An, 3]`, with the last dimension
        storing the position where the camera is looking at.
      camera_up: A Tensor of shape `[A1, ..., An, 3]`, where the last dimension
        defines the up vector of the camera.
      field_of_view:  A Tensor of shape `[A1, ..., An, 1]`, where the last
        dimension represents the vertical field of view of the frustum expressed
        in radians. Note that values for `vertical_field_of_view` must be in the
        range (0,pi).
      image_size: A tuple of shape `(W,H)` where `W` and `H` are respectively
        the width and height (in pixels) of the rasterized image.
      near_plane: A Tensor of shape `[A1, ..., An, 1]`, where the last dimension
        captures the distance between the viewer and the near clipping plane.
        Note that values for `near_plane` must be non-negative.
      far_plane: A Tensor of shape `[A1, ..., An, 1]`, where the last dimension
        captures the distance between the viewer and the far clipping plane.
        Note that values for `far_plane` must be non-negative.
      bottom_left: A Tensor of shape `[A1, ..., An, 2]`, where the last
        dimension captures the position (in pixels) of the lower left corner of
        the screen.
      name: A name for this op. Defaults to 'triangle_rasterizer_init'.
    """
    with tf.compat.v1.name_scope(name, "triangle_rasterizer_init", [
        background_geometry, background_attribute, camera_origin, look_at,
        camera_up, field_of_view, near_plane, far_plane, bottom_left
    ]):
      self.background_geometry = tf.convert_to_tensor(value=background_geometry)
      self.background_attribute = tf.convert_to_tensor(
          value=background_attribute)
      self.camera_origin = tf.convert_to_tensor(value=camera_origin)
      self.look_at = tf.convert_to_tensor(value=look_at)
      self.camera_up = tf.convert_to_tensor(value=camera_up)
      self.field_of_view = tf.convert_to_tensor(value=field_of_view)
      self.image_size_float = image_size
      self.image_size_int = (int(image_size[0]), int(image_size[1]))
      self.near_plane = tf.convert_to_tensor(value=near_plane)
      self.far_plane = tf.convert_to_tensor(value=far_plane)
      self.bottom_left = tf.convert_to_tensor(value=bottom_left)

      shape.check_static(
          tensor=self.background_geometry,
          tensor_name="background_geometry",
          has_rank=3,
          has_dim_equals=((-2, 3), (-1, 3)))
      shape.check_static(
          tensor=self.background_attribute,
          tensor_name="background_attribute",
          has_rank=3,
          has_dim_equals=(-2, 3))
      shape.compare_batch_dimensions(
          tensors=(self.background_geometry, self.background_attribute),
          last_axes=-3,
          tensor_names=("background_geometry", "background_attribute"),
          broadcast_compatible=False)

      # Construct the pixel grid.
      width = self.image_size_float[0]
      height = self.image_size_float[1]
      px = tf.linspace(0.5, width - 0.5, num=int(width))
      py = tf.linspace(0.5, height - 0.5, num=int(height))
      xv, yv = tf.meshgrid(px, py)
      self.pixel_position = tf.stack((xv, yv), axis=-1)

      # Construct the view projection matrix.
      world_to_camera = glm.look_at_right_handed(camera_origin, look_at,
                                                 camera_up)
      perspective_matrix = glm.perspective_right_handed(field_of_view,
                                                        (width / height,),
                                                        near_plane, far_plane)
      perspective_matrix = tf.squeeze(perspective_matrix)
      self.view_projection_matrix = tf.linalg.matmul(perspective_matrix,
                                                     world_to_camera)

  def rasterize(self, scene_geometry=None, scene_attributes=None, name=None):
    """Rasterizes the scene.

    This rasterizer estimates which triangle is associated to each pixel using
    OpenGL. Then the value of attributes are estimated using Tensorflow,
    allowing to get gradients flowing through the attributes. Attributes can be
    depth, appearance, or more generally, any K-dimensional representation. Note
    that similarly to algorithms like Iterative Closest Point (ICP), not having
    gradients through correspondance does not prevent from optimizing the scene
    geometry. Custom gradients can be defined to alleviate this property.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      scene_geometry: A tensor of shape `[A1, ..., An, T, 3, 3]` containing
        batches of `T` triangles, each defined by a triplet of 3D vertices.
      scene_attributes: A tensor of shape `[A1, ..., An, T, 3, K]` containing
        batches of `T` triangles, with their vertices associated with
        K-dimensional attributes.
      name: A name for this op. Defaults to 'triangle_rasterizer_rasterize'.

    Returns:
      A tensor of shape `[A1, ..., An, W, H, K]` containing batches of images of
      width `W` and height `H`, where each pixel contains attributes rasterized
      from the scene.
    """
    with tf.compat.v1.name_scope(name, "triangle_rasterizer_rasterize",
                                 [scene_geometry, scene_attributes]):
      scene_geometry = tf.convert_to_tensor(value=scene_geometry)
      scene_attributes = tf.convert_to_tensor(value=scene_attributes)

      shape.check_static(
          tensor=scene_geometry,
          tensor_name="scene_geometry",
          has_rank_greater_than=2,
          has_dim_equals=((-2, 3), (-1, 3)))
      shape.check_static(
          tensor=scene_attributes,
          tensor_name="scene_attributes",
          has_rank_greater_than=2,
          has_dim_equals=(-2, 3))
      shape.compare_batch_dimensions(
          tensors=(scene_geometry, scene_attributes),
          last_axes=-3,
          tensor_names=("scene_geometry", "scene_attributes"),
          broadcast_compatible=False)

      batch_shape = scene_geometry.shape[:-3]

      def dim_value(dim):
        return 1 if dim is None else tf.compat.v1.dimension_value(dim)

      batch_shape = [dim_value(dim) for dim in batch_shape]

      background_geometry = tf.broadcast_to(
          self.background_geometry,
          batch_shape + self.background_geometry.shape)
      background_attribute = tf.broadcast_to(
          self.background_attribute,
          batch_shape + self.background_attribute.shape)
      geometry = tf.concat((background_geometry, scene_geometry), axis=-3)
      attributes = tf.concat((background_attribute, scene_attributes), axis=-3)

      view_projection_matrix = tf.broadcast_to(
          input=self.view_projection_matrix,
          shape=batch_shape + self.view_projection_matrix.shape)
      rasterized_face = render_ops.rasterize(
          num_points=geometry.shape[-3],
          variable_names=["view_projection_matrix", "triangular_mesh"],
          variable_kinds=["mat", "buffer"],
          variable_values=[
              view_projection_matrix,
              # batch_shape has to be explicitly converted in int32 as it is
              # float if there is no batch.
              tf.reshape(
                  geometry,
                  shape=tf.concat(
                      values=(tf.convert_to_tensor(
                          value=batch_shape, dtype=tf.int32), (-1,)),
                      axis=0))
          ],
          output_resolution=self.image_size_int,
          vertex_shader=vertex_shader,
          geometry_shader=geometry_shader,
          fragment_shader=fragment_shader)
      triangle_index = tf.dtypes.cast(rasterized_face[..., 2], tf.int32)
      vertices_per_pixel = tf.gather(
          geometry, triangle_index, axis=-3, batch_dims=len(batch_shape))
      attributes_per_pixel = tf.gather(
          attributes, triangle_index, axis=-3, batch_dims=len(batch_shape))
      return glm.perspective_correct_interpolation(
          vertices_per_pixel, attributes_per_pixel, self.pixel_position,
          self.camera_origin, self.look_at, self.camera_up, self.field_of_view,
          self.image_size_float, self.near_plane, self.far_plane,
          self.bottom_left)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()

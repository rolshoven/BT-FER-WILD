import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class Transformer(Layer):
    """ This layer is an adaption of the code provided in the following repository:
    https://github.com/kevinzakka/spatial-transformer-network/
    """
    def __init__(self, output_dim=(192, 192), **kwargs):
        self.output_dim = output_dim
        super(Transformer, self).__init__(**kwargs)

    def call(self, tensors):
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        fmap, theta = tensors
        return self._spatial_transformer_network(fmap, theta, self.output_dim)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        height, width = self.output_size
        num_batches = input_shape[0][0]
        num_channels = input_shape[0][-1]
        return (num_batches, height, width, num_channels)

    def _spatial_transformer_network(self, input_fmap, theta, out_dims=None, **kwargs):
        """
         Spatial Transformer Network layer implementation as described in [1].
         The layer is composed of 3 elements:
         - localization_net: takes the original image as input and outputs
           the parameters of the affine transformation that should be applied
           to the input image.
         - affine_grid_generator: generates a grid of (x,y) coordinates that
           correspond to a set of points where the input should be sampled
           to produce the transformed output.
         - bilinear_sampler: takes as input the original image and the grid
           and produces the output image using bilinear interpolation.
         Input
         -----
         - input_fmap: output of the previous layer. Can be input if spatial
           transformer layer is at the beginning of architecture. Should be
           a tensor of shape (B, H, W, C).
         - theta: affine transform tensor of shape (B, 6). Permits cropping,
           translation and isotropic scaling. Initialize to identity matrix.
           It is the output of the localization network.
         Returns
         -------
         - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
         Notes
         -----
         [1]: 'Spatial Transformer Networks', Jaderberg et. al,
              (https://arxiv.org/abs/1506.02025)
     """

        # grab input dimensions
        B = tf.shape(input_fmap)[0]
        H = tf.shape(input_fmap)[1]
        W = tf.shape(input_fmap)[2]

        # reshape theta to (B, 2, 3)
        theta = tf.reshape(theta, [B, 2, 3])

        # generate grids of same size or upsample/downsample if specified
        if out_dims:
            out_H = out_dims[0]
            out_W = out_dims[1]
            batch_grids = self._affine_grid_generator(out_H, out_W, theta)
        else:
            batch_grids = self._affine_grid_generator(H, W, theta)

        x_s = batch_grids[:, 0, :, :]
        y_s = batch_grids[:, 1, :, :]

        # sample input with grid to get output
        out_fmap = self._bilinear_sampler(input_fmap, x_s, y_s)

        return out_fmap

    def _get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    def _affine_grid_generator(self, height, width, theta):
        """
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.
        Input
        -----
        - height: desired height of grid/output. Used
          to downsample or upsample.
        - width: desired width of grid/output. Used
         to downsample or upsample.
        - theta: affine transform matrices of shape (num_batch, 2, 3).
          For each image in the batch, we have 6 theta parameters of
          the form (2x3) that define the affine transformation T.
        Returns
        -------
        - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
          The 2nd dimension has 2 components: (x, y) which are the
          sampling points of the original image for each point in the
          target image.
        Note
        ----
        [1]: the affine transformation allows cropping, translation,
              and isotropic scaling.
        """
        num_batch = tf.shape(theta)[0]
        # create normalized 2D grid
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid num_batch times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

        return batch_grids

    def _bilinear_sampler(self, img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.
        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.
        Input
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator.
        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1/H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = self._get_pixel_value(img, x0, y0)
        Ib = self._get_pixel_value(img, x0, y1)
        Ic = self._get_pixel_value(img, x1, y0)
        Id = self._get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return out


class IslandLossLayer(Layer):

    def __init__(self, feature_dim, num_classes=8, alpha=0.5, balance=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.balance = balance

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_classes, self.feature_dim),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is N x feature_dim, x[1] is N x num_classes onehot, self.centers is num_classes x feature_dim
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # num_classes x feature_dim
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # num_classes x 1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        center_loss = x[0] - K.dot(x[1], self.centers)
        center_loss = K.sum(self.result ** 2, axis=1, keepdims=True)  # / K.dot(x[1], center_counts)

        pair_dist = K.dot(K.transpose(self.centers), self.centers)
        pair_dist = pair_dist - K.dot(self.centers, self.centers)
        pair_dist = pair_dist / K.sqrt(K.square(pair_dist))
        pair_dist = K.sum(pair_dist, keepdims=True)

        self.result = center_loss - pair_dist
        return self.result  # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


class GlobalCovPooling2D(Layer):
    """ Covariance pooling operation for spatial data.

    Adapted from https://github.com/jiangtaoxie/fast-MPN-COV.

    # Arguments
        num_iter: An integer, number of Newton-Schulz iterations
    # Input shape
        4D tensor with shape:
        `(batch_size, rows, cols, channels)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels * (channels + 1) / 2)`
    """

    def __init__(self, num_iter=3, **kwargs):
        self.num_iter = num_iter
        super(GlobalCovPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GlobalCovPooling2D, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        num_rows = K.int_shape(inputs)[1]
        num_cols = K.int_shape(inputs)[2]
        num_channels = K.int_shape(inputs)[3]
        n = num_rows * num_cols
        X = K.reshape(inputs, (batch_size, num_channels, n))
        factor = K.cast(1 / n, K.floatx())
        I_hat = factor * (K.eye(n) - factor * K.ones((n, n)))
        I_hat = K.tile(K.expand_dims(I_hat, axis=0), (batch_size, 1, 1))  # One identity matrix per sample in batch
        Sigma = K.batch_dot(K.batch_dot(X, I_hat), K.permute_dimensions(X, (0, 2, 1)))

        # Pre-normalization
        trace = K.sum(K.sum(K.eye(num_channels) * Sigma, axis=1, keepdims=True), axis=2, keepdims=True)
        A = Sigma / trace

        # Newton-Schulz Iteration
        Y = A
        Z = K.eye(num_channels)
        Z = K.tile(K.expand_dims(Z, axis=0), (batch_size, 1, 1))
        I3 = 3 * K.eye(num_channels)
        I3 = K.tile(K.expand_dims(I3, axis=0), (batch_size, 1, 1))
        for i in range(self.num_iter):
            Y = 0.5 * K.batch_dot(Y, I3 - K.batch_dot(Z, Y))
            Z = 0.5 * K.batch_dot(I3 - K.batch_dot(Z, Y), Z)

        # Post-compensation
        C = K.sqrt(trace) * Y

        # Extract upper triangular matrix as vector
        ones = K.ones((num_channels, num_channels))
        mask = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask = K.cast(mask, 'bool')  # Convert integer mask to boolean mask
        triuvec = tf.boolean_mask(C, mask, axis=1)  # Apply mask to 2nd and 3rd dimension
        triuvec.set_shape((None, num_channels * (num_channels + 1) / 2))  # Set correct shape manually

        return triuvec

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_channels = input_shape[-1]
        return (batch_size, int(num_channels * (num_channels + 1) / 2))

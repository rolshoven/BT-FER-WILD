from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import categorical_crossentropy

def island_crossentropy_loss(features, balance=0.01, num_classes=8):
    """Calculates the island loss

    Args:
    features -- feature vector in one-hot encoding of shape (batch_size, num_fts)
    balance -- weight term for the pairwise distance term that
               is added to the center loss.

    """
    def custom_loss(y_true, y_pred):
        """Args: y_true -- label vector of shape (batch_size, num_classes)"""
        samples_per_cluster = K.transpose(K.sum(y_true, axis=0, keepdims=True) + 1) # Add 1 to avoid division by zero
        centers = K.dot(K.transpose(y_true), features) / samples_per_cluster
        center_loss = 0.5 * K.sum(K.square(features - K.dot(y_true, centers)))

        center_dot_combinations = K.dot(centers, K.transpose(centers))
        center_dot_combinations_normed = K.sqrt(K.square(center_dot_combinations))
        pair_dist = center_dot_combinations / center_dot_combinations_normed
        # subtract diagonal of pair_dist which only contains ones
        pair_dist = pair_dist - K.eye(num_classes)
        pair_dist = pair_dist + 1
        pair_dist = K.sum(pair_dist)

        island_loss = center_loss + pair_dist

        return categorical_crossentropy(y_true, y_pred) + island_loss
    return custom_loss

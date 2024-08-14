import tensorflow as tf
from tensorflow.keras import backend

def focal_loss(gamma=2., alpha=0.25, label_smoothing=0.1):
    """
    Computes the focal loss, which is designed to address the class imbalance problem by down-weighting well-classified examples.

    Args:
        gamma (float, optional): The focusing parameter that adjusts the rate at which easy examples are down-weighted. A higher value of `gamma` reduces the relative loss for well-classified examples (p > 0.5), focusing more on hard examples. Defaults to 2.0.
        alpha (float, optional): The balancing factor for the classes. Typically used to balance the importance of positive and negative examples. Defaults to 0.25.
    
    Returns:
        function: A loss function that computes the focal loss for a given pair of true and predicted labels.
    """

    def focal_loss_fixed(y_true, y_pred):
        epsilon = backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Apply label smoothing
        if label_smoothing > 0:
            num_classes = tf.shape(y_true)[-1]
            y_true = (1.0 - label_smoothing) * y_true + label_smoothing / tf.cast(num_classes, tf.float32)
        
        y_pred = backend.clip(y_pred, epsilon, 1. - epsilon)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        fl = - alpha_t * backend.pow((1 - p_t), gamma) * backend.log(p_t)
        
        return backend.mean(backend.sum(fl, axis=-1))
    
    return focal_loss_fixed
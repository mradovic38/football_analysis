import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError(reduction="none")
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

def visibility_based_mse(lmbda=0.5):
  mse = tf.keras.losses.MeanSquaredError(reduction="none")
  binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

  def loss(y_true, y_pred):
    visibility_mask = y_true[:, :, 2]

    y_true_points = y_true[:, :, :2]
    y_pred_points = y_pred[:, :, :2]
    
    # Calculate MSE for all points
    point_loss = mse(y_true_points, y_pred_points)
    
    # Apply visibility mask to point loss
    masked_point_loss = point_loss * tf.expand_dims(visibility_mask, axis=-1)
    
    # Calculate mean of masked point loss
    mean_point_loss = tf.reduce_mean(masked_point_loss)
    
    # Calculate binary cross-entropy for visibility mask
    visibility_loss = binary_crossentropy(y_true[:, :, 2], visibility_mask)
    
    # Combine point loss and visibility loss
    combined_loss = lmbda * mean_point_loss + (1-lmbda) * tf.reduce_mean(visibility_loss)
    
    return combined_loss

  return loss
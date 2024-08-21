import matplotlib.pyplot as plt

def plot_examples(dataset, num_examples=1, normalized=True):

  for batch in dataset.take(1):
    batch_images, batch_keypoints = batch
    for image, keypoints in zip(batch_images[:num_examples], batch_keypoints[:num_examples]):

      plt.figure(figsize=(12,7))
      plt.imshow(image.numpy().astype('uint8'))
      plt.axis(False)


      # Reshape the keypoints array to have shape (num_keypoints, 3)
      keypoints = keypoints.numpy()
      keypoints = keypoints.reshape(-1, 3)
      for i, keypoint in enumerate(keypoints):
        x, y, visibility = keypoint
        if visibility == 2:  # Only plot visible keypoints
          plt.scatter(x * (image.shape[0] if normalized else 1), 
                      y * (image.shape[1] if normalized else 1), 
                      s=50, c='red', marker='o')
          plt.text(x * (image.shape[0] if normalized else 1), 
                   y * (image.shape[1] if normalized else 1), str(i), 
                   fontsize=12, color='blue', ha='right', va='bottom')

      plt.show()
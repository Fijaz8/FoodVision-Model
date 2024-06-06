
import tensorflow as tf
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
   """
    Reads in an image from filename, turns it into a tensor and reshapes it to (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default is 224
    scale (bool): whether to scale pixel values to range(0, 1), default is True

    Returns
    -------
    Tensor: Image tensor of shape (img_shape, img_shape, 3)
    """
  # Convert the numpy array to a tensor
  img = tf.convert_to_tensor(filename,dtype=tf.float32)

  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])

  # Scale the image pixel values to the range (0, 1) if scale is True
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

 # List of class names for prediction
class_names = [
    'Biryani', 'Chole-Bhature', 'Jalebi', 'Kofta', 'Naan',
    'Paneer-Tikka', 'Pani-Puri', 'Pav-Bhaji', 'Vadapav',
    'Dabeli', 'Dal', 'Dhokla', 'Dosa', 'Kathi', 'Pakora'
]
def predict_data(model, filename, class_names):
  """
    Imports an image located at filename, makes a prediction on it with a trained model,
    and returns the predicted class.

    Parameters
    ----------
    model: Trained TensorFlow/Keras model
    filename (str): Path to the image file
    class_names (list): List of class names corresponding to model output

    Returns
    -------
    str: Predicted class name
    """
  # Load and preprocess the image
  img = load_and_prep_image(filename)

  # Make a prediction on the preprocessed image
  pred_prob = model.predict(tf.expand_dims(img, axis=0)) # make prediction on image with shape [None, 224, 224, 3]

  # Get the predicted class with the highest probability
  pred_class = class_names[pred_prob.argmax()]

  return pred_class


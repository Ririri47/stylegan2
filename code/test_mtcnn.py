from mtcnn import MTCNN
import tensorflow as tf
import numpy as np

def load_image(image, dtype=tf.float32, device="CPU:0"):
    """
    Loads an image and decodes it into a TensorFlow tensor, optionally normalizing it.

    Args:
        image (str, np.ndarray, or tf.Tensor): The input image. It can be:
                                               - A file path to the image as a string.
                                               - A TensorFlow tensor or a NumPy array representing an image.
        dtype (tf.DType, optional): The desired data type for the decoded image. Default is tf.float32.
        device (str, optional): the target device for the operation. Using CPU most of the times should be fine.
    
    Returns:
        np.ndarray: The decoded image array, with shape (height, width, channels) and dtype `dtype`. If 
                   `normalize=True`, the image values will be scaled to the range [0, 1].
    """

    with tf.device(device):
        is_tensor = tf.is_tensor(image) or isinstance(image, np.ndarray)

        if is_tensor:
            decoded_image = image
            if isinstance(decoded_image, np.ndarray):
                return decoded_image
            # If it's a TensorFlow tensor, convert to numpy
            with tf.Session() as sess:
                decoded_image = sess.run(decoded_image)
        else:
            try:
                if isinstance(image, str):
                    image_data = tf.io.read_file(image)  # Read image from file
                else:
                    image_data = image  # Assume image data is provided directly
            except tf.errors.NotFoundError:
                image_data = image  # If file not found, use the input directly

            # Decode the image with 3 channels (RGB)
            # TensorFlow 1.x: decode_image doesn't support dtype parameter
            decoded_image_tensor = tf.image.decode_image(image_data, channels=3)
            # Convert to numpy array using session
            with tf.Session() as sess:
                decoded_image = sess.run(decoded_image_tensor)

        # If dtype is float, adjust the image scale
        if dtype in [tf.float16, tf.float32]:
            decoded_image = decoded_image.astype(np.float32)
            decoded_image /= 255.0  # Normalize to [0, 1] range
        else:
            decoded_image = decoded_image.astype(np.uint8)

    return decoded_image

# Create a detector instance
detector = MTCNN()

# Load an image
image = load_image("../imgs/00018.png")

# Detect faces in the image
result = detector.detect_faces(image)

# Display the result
print(result)
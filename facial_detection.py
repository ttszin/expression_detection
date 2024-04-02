import cv2
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())


from deepface import DeepFace

img = cv2.imread('images.jpg')

plt.imshow(img[:, :, : : -1])

plt.show()

result = DeepFace.analyze(img, actions=['emotion'])

print(result)
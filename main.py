from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
#from matplotlib import pyplot as plt
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

cur = os.getcwd()

mtcnn = MTCNN(image_size=160, margin=25, device='cuda')
model = InceptionResnetV1(pretrained='vggface2').eval()
# model = InceptionResnetV1(pretrained='casia-webface').eval()

img = Image.open(cur + '/PDI/test_images/sample.jpg')

# Get cropped and prewhitened image tensor
face = mtcnn(img)#, save_path=cur + '/PDI/saved_files/sample3.jpg')
#img_cropped.shape  # ([3, 160, 160])

# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
model.classify = True
img_probs = model(face.unsqueeze(0))
print(img_probs)  # Tensor com características da face
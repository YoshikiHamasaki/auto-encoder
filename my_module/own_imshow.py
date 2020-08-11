import torchvision
import numpy as np 
import matplotlib.pyplot as plt

def own_imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    npimg_reshape = np.transpose(npimg,(1,2,0)) 
    plt.imshow(npimg_reshape)
    plt.show()

    return npimg_reshape

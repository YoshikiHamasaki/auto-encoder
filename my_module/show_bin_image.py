import numpy as np
import matplotlib.pyplot as plt
from my_module import reconstruction_error as re
from torch.autograd import Variable

def show_image(test_loader,test_img_index,model):

    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    test_img = images[test_img_index]
    in_img = test_img.reshape((28,28))
    in_img = in_img.detach().numpy()
    in_img = in_img/2+ 0.5
    
    out_img = test_img.view(test_img.size(0),-1)
    out_img= Variable(out_img)
    pred = model(out_img)
    pred = pred.detach().numpy()
    pred = pred/2 + 0.5
    pred = pred.reshape((28,28))
    
    #print(pred)
    #print(in_img)
    
    r_error = re.reconstruction_error(in_img,pred)
    print(r_error)
    
    plt.subplot(2,1,1)
    plt.imshow(in_img, cmap = "gray", vmin =0,vmax=1)
    
    plt.subplot(2,1,2)
    plt.imshow(pred, cmap = "gray", vmin = 0, vmax = 1)
    
    plt.show()

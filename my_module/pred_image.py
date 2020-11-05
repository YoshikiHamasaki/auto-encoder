import numpy as np
import matplotlib.pyplot as plt
from my_module import reconstruction_error as recon
from torch.autograd import Variable
from my_module import own_imshow as own_imshow
from my_module.to_model import to_model

def pred_bin_image(test_loader,test_img_index,model):

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
    
    r_error = recon.reconstruction_error(in_img,pred)
    print(r_error)
    
    plt.subplot(2,1,1)
    plt.imshow(in_img, cmap = "gray", vmin =0,vmax=1)
    
    plt.subplot(2,1,2)
    plt.imshow(pred, cmap = "gray", vmin = 0, vmax = 1)
    
    plt.show()


def pred_color_image(test_loader,test_img_index,model,input_size): 

    iterator = iter(test_loader)
    ori_img, _ = next(iterator)

    if test_img_index == "ALL":
        input_img = ori_img

    else:
        input_img = ori_img[test_img_index]

    
    recon_in = own_imshow.own_imshow(input_img)
    #print(input_img.shape)
    
    result_img = input_img.view(-1,input_size)
    result_img = Variable(result_img)
    #print(result_img.shape)
    result_img = model(result_img)
    
    result_img = result_img.reshape(-1,3,28,28)

    recon_out = own_imshow.own_imshow(result_img)

    if not test_img_index == "ALL":
        r_error = recon.color_reconstruction_error(recon_in,recon_out)
        print(r_error)


def pred_lab_image(test_loader,test_img_index,model,input_size): 

    iterator = iter(test_loader)
    ori_img, _ = next(iterator)

    if test_img_index == "ALL":
        input_img = ori_img
        to_model(input_img,len(input_img),input_size,model)

    else:
        input_img = ori_img[test_img_index]
        to_model(input_img,test_img_index,input_size,model)

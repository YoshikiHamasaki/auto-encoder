from my_module import make_data_set as mds
from my_module import autoencoder as AE
from my_module import calculate as cal
from my_module import pred_image as pred
from my_module import own_imshow as own_imshow
from my_module import select_optimizer_type as select_optim
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from torch.autograd import Variable



def for_color_detect(num_epochs,optimizer_type,learning_rate,input_size,train_csv_path,test_csv_path,train_image_path,test_image_path):

    device =  "cpu" 
    save_type = "color"
    
    train_loader, test_loader = mds.color_make_data_set(
            train_csv_path,test_csv_path,train_image_path,test_image_path) 
    model = AE.color_Autoencoder(input_size) 

    optimizer = select_optim.select_optimizer_type(optimizer_type,model,learning_rate)

    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/{save_type}_{optimizer_type}_lr_{learning_rate}_epoch_{num_epochs}_model.pkl") 
    np.save(f"./result/loss_list.npy", np.array(loss_list)) 
    iterator = iter(test_loader)
    input_img, _ = next(iterator)
    own_imshow.own_imshow(input_img)
    plt.show()
    
    result_img = input_img.view(input_img.size(0),-1)
    result_img = Variable(result_img)
    pred_img = model(result_img)
    
    pred = pred_img.reshape(-1,3,28,28)
    own_imshow.own_imshow(pred)

def for_bin_detect(num_epochs,optimizer_type,learning_rate,weight_decay,input_size,train_csv_path,test_csv_path,train_image_path,test_image_path):

    device =  'cpu'
    save_type = "binary"
    
    train_loader, test_loader = mds.bin_make_data_set(
            train_csv_path,test_csv_path,train_image_path,test_image_path) 
    model = AE.bin_Autoencoder()
    optimizer = select_optim.select_optimizer_type(optimizer_type,model,learning_rate,weight_decay)
    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/{save_type}_{optimizer_type}_lr_{learning_rate}_epoch_{num_epochs}_model.pkl") 
    np.save(f"./result/loss_list.npy", np.array(loss_list)) 
    pred.pred_bin_image(test_loader,3,model) 
    #loss_list = np.load('{}/loss_list.npy'.format(out_dir))
    #plt.plot(loss_list)
    #plt.xlabel('iteration')
    #plt.ylabel('loss')
    #plt.grid()
    #plt.show()

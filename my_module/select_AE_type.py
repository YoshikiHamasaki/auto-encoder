from my_module import autoencoder as AE

def select_AE_type(AE_type,input_size):
    if AE_type == "COLOR":
        model = AE.color_autoencoder(input_size)

    elif AE_type == "BIN":
        model = AE.bin_autoencoder()
    
    elif AE_type == "LAB":
        model = AE.lab_autoencoder(input_size)
    
    elif AE_type == "CONV":
        model = AE.conv_autoencoder()
    
    else:
        print("select AE_type error")
    return model


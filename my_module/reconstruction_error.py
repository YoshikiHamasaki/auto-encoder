def reconstruction_error(input_img,output_img):  
    save = []
    for y in range(input_img.shape[0]):
        for x in range(input_img.shape[1]):
            e = (input_img[y][x]-output_img[y][x])**2
            save.append(e)

    E = sum(save)/float(len(save))
    return E

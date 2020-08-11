def bin_reconstruction_error(input_img,output_img):  
    save = []
    for y in range(input_img.shape[0]):
        for x in range(input_img.shape[1]):
            e = (input_img[y][x]-output_img[y][x])**2
            save.append(e)

    E = sum(save)/float(len(save))
    return E

def color_reconstruction_error(input_img,output_img):
    save = []
    for c in range(input_img.shape[2]):
        for y in range(input_img.shape[0]):
            for x in range(input_img.shape[1]):
                e = ((input_img[y][x][c]-output_img[y][x][c])**2)*100
            save.append(e)

    E = sum(save)/float(len(save))
    return E
    

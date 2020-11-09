import math

def reconstruction_error_2d(input_img,output_img,error_th):  
    save = []
    for y in range(input_img.shape[0]):
        for x in range(input_img.shape[1]):
            e = math.sqrt((input_img[y][x]-output_img[y][x])**2)
            save.append(e)

    E = sum(save)/float(len(save))
    if error_th = "40" and E < 4.0:
        print("normal pipe image")
    elif error_th = "40" and E > 4.0:
        print("abnormal pipe image")
     
    if error_th = "60" and E < 4.0:
        print("normal pipe image")
    elif error_th = "60" and E > 4.0:
        print("abnormal pipe image")
    
    if error_th = "100" and E < 4.0:
        print("normal pipe image")
    elif error_th = "100" and E > 4.0:
        print("abnormal pipe image")
    
    if error_th = "color" and E < 4.0:
        print("normal pipe image")
    elif error_th = "color" and E > 4.0:
        print("abnormal pipe image")
    
    return E



def reconstruction_error_3d(input_img,output_img,error_th):
    save = []
    for c in range(input_img.shape[2]):
        for y in range(input_img.shape[0]):
            for x in range(input_img.shape[1]):
                e = math.sqrt((input_img[y][x][c]-output_img[y][x][c])**2)*100
            save.append(e)

    E = sum(save)/float(len(save))

    if error_th = "40" and E < 4.0:
        print("normal pipe image")
    elif error_th = "40" and E > 4.0:
        print("abnormal pipe image")
     
    if error_th = "60" and E < 4.0:
        print("normal pipe image")
    elif error_th = "60" and E > 4.0:
        print("abnormal pipe image")
    
    if error_th = "100" and E < 4.0:
        print("normal pipe image")
    elif error_th = "100" and E > 4.0:
        print("abnormal pipe image")
    
    if error_th = "color" and E < 4.0:
        print("normal pipe image")
    elif error_th = "color" and E > 4.0:
        print("abnormal pipe image")

    return E

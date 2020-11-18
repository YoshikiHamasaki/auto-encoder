# -*- coding:utf-8 -*-
import cv2
import pygame
from pygame.locals import *
import sys
import math
import numpy
import glob
from natsort import natsorted
import shutil
import datetime
import os

date_now = datetime.datetime.now()
date_month = date_now.month
date_day = date_now.day
date_hour = date_now.hour
date_min = date_now.minute


####### setting parameter ############
path = '../image/image-data/'
image_path = natsorted(glob.glob(path + '/*.jpg'))
out_path = f"../image/image-data/trimming/{date_month}_{date_day}_{date_hour}_{date_min}"
os.mkdir(out_path)
######################################





def get_image_point(path):
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("get_point")
    bg = pygame.image.load(path)
    bg = pygame.transform.scale(bg, (w, h))
    rect_bg = bg.get_rect()
    x, y = 0, 0
    mx, my = 0, 0
    x_s = 100 

    while (1):
        pygame.display.update()     # 画面を更新
        pygame.time.wait(30)
        screen.fill((0,0,0))        # 画面を黒色(#000000)に塗りつぶし
        screen.blit(bg, rect_bg)
        pygame.draw.line(screen, (255,0,0), (x, y), (mx, y),2)
        pygame.draw.line(screen, (255,0,0), (x, y), (x, my),2)
        pygame.draw.line(screen, (255,0,0), (mx, y), (mx, my),2)
        pygame.draw.line(screen, (255,0,0), (x, my), (mx, my),2)

        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                pygame.mouse.set_pos([100, 100])
            if event.type == MOUSEMOTION:
                mx, my = event.pos
            
            if event.type == KEYDOWN:       # キーを押したとき
                if event.key == K_ESCAPE:   # Escキーが押されたとき
                    pygame.quit()
                    sys.exit()

                if event.key == K_s:
                    x_s = x
                    y_s = y
                    mx_s = mx
                    my_s = my
            
            if event.type == QUIT:  # 閉じるボタンが押されたら終了
                pygame.quit()       # Pygameの終了(画面閉じられる)
                sys.exit()
        if x_s == x:
            break
    return x_s, y_s, mx_s, my_s

def image_trimming(path, number,out_path):
    img = cv2.imread(path, 1)
    img_h, img_w, img_c = img.shape
    h_ratio = img_h / h
    w_ratio = img_w / w
    img_x = x * w_ratio
    img_mx = mx * w_ratio
    img_y = y * h_ratio
    img_my = my * h_ratio
    dx = img_mx - img_x
    dy = img_my - img_y
    count = 0
    for l in range(int(dx/28)):
        for m in range(int(dy/28)):
            img2 = img[int(img_y + m*28):int(img_y + (m+1)*28), int(img_x + l*28):int(img_x + (l+1)*28)]
            count += 1
    #img2 = img[int(img_y):int(img_my), int(img_x):int(img_mx)]
            cv2.imwrite(f'{out_path}/tri_{number}_{count}.jpg', img2)

for i,f in enumerate(image_path):
    im = cv2.imread(image_path[i],1)
    im_h, im_w, im_c = im.shape
    if im_h < im_w:
        (w, h) = (800, 600)
    elif im_w < im_h:
        (w, h) = (600, 800)
    else:
        (w, h) = (600, 600)
    x, y, mx, my = get_image_point(image_path[i])
    a,b = os.path.splitext(os.path.basename(image_path[i]))
    image_trimming(image_path[i],a,out_path)

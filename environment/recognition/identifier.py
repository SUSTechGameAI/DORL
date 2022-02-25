import gym_gvgai as gvg
import time
import numpy as np
import json
import os
from PIL import Image
elm_size=16
root = os.path.dirname(os.path.abspath(__file__))

def buildIdentifier(game, lvls):
    axisx = np.array([4,4,4,4,4,1,3,5,7,9])
    axisy = np.array([0,2,4,6,8,5,5,5,5,5])
    main = {}
    visit = {}
    total = 1
    for l in lvls:
        # print(l)
        with open(root+'/level_txt/%s/%s.txt' % (game, l), 'r') as f:
            lvl = str.split(f.read(), '\n')
        env = gvg.make('gvgai-%s-%s-v0' % (game, l))
        pixels = env.reset()
        env.render()
        h, w = len(pixels)//10, len(pixels[0])//10
        for i in range(h):
            for j in range(w):
                t = lvl[i][j]
                if t not in visit:
                    if t == 'A':
                        main['A'] = elm_size-1
                        visit[t] = elm_size-1
                    else:
                        visit[t] = total
                        # print(t,total)
                        total += 1
                xs = axisx + np.array([i*10]*10)
                ys = axisy + np.array([j*10]*10)
                #val = ( sum(pixels[xs, ys, 0])+sum(pixels[xs, ys, 1])+sum(pixels[xs, ys, 2]))//25
                keyv=''
                for k in range(10):
                    tmp=0
                    for l in range(3):
                        tmp+=pixels[xs[k],ys[k],l]
                    keyv+=str(tmp//3)
                    if k<9:keyv+=','
                # if(t=='A'):
                #     print(pixels[xs, ys, 0],pixels[xs, ys, 1],pixels[xs, ys, 2])
                #     print(keyv)
                main[keyv] = visit[t]
    with open(root+"/here.json", 'w') as file:
        json.dump(main, file)
    # print(main)
class Identifier:
    axisx = np.array([4,4,4,4,4,1,3,5,7,9])
    axisy = np.array([0,2,4,6,8,5,5,5,5,5])
    def __init__(self,img, fileName):
        self.main = {}
        self.elm = []
        self.val = []
        self.playerx=0
        self.playery=0
        self.h = img.shape[0]//10
        self.w = img.shape[1]//10
        # print(os.getcwd())
        with open(fileName, 'r') as file:
            self.main = json.load(file)
        for key in self.main:
            if key=='A':continue
            val = key.split(',')
            tmp=np.empty(10)
            for i in range(10):
                tmp[i]=int(val[i])
            mat = np.empty((self.h,self.w,10))
            mat[:,:]=tmp
            self.elm.append(mat)
            self.val.append(self.main[key])

    def identify(self, img):
        # start = time.process_time_ns()
        # img = np.array(img)
        # print(np.array(img).shape)
        h, w = img.shape[0] // 10, img.shape[1] // 10
        elm_mat = np.empty((h, w, 10), dtype=int)
        for i in range(h):
            for j in range(w):
                xs = self.axisx + np.array([i * 10] * 10)
                ys = self.axisy + np.array([j * 10] * 10)
                elm_mat[i][j] = np.sum(img[xs, ys, 0:3], axis=1)
        elm_mat = elm_mat // 3
        dif_mat = np.empty((h, w, len(self.elm)))
        for i in range(len(self.elm)):
            dif_mat[:, :, i] = np.sum(abs(elm_mat - self.elm[i]), axis=2)
        res = np.argmin(dif_mat, axis=2)
        for i in range(h):
            for j in range(w):
                res[i][j] = self.val[res[i][j]]
                if 11 <= res[i][j] <= 15:
                    self.playerx = i + h
                    self.playery = j + w
        res = np.lib.pad(res, ((h, h), (w, w)), 'constant', constant_values=0)
        img_extend = np.lib.pad(img, ((h * 10, h * 10), (w * 10, w * 10), (0, 0)), 'constant', constant_values=0)
        mat_go = res[self.playerx - h:self.playerx + h + 1, self.playery - w:self.playery + w + 1]
        mat_lo = res[self.playerx - 2:self.playerx + 3, self.playery - 2:self.playery + 3]
        mat_go = np.eye(elm_size)[mat_go]
        mat_lo = np.eye(elm_size)[mat_lo]
        img_go = img_extend[(self.playerx - h) * 10: (self.playerx + h + 1) * 10,
                 (self.playery - w) * 10: (self.playery + w + 1) * 10]
        img_lo = img_extend[(self.playerx - 2) * 10: (self.playerx + 3) * 10,
                 (self.playery - 2) * 10: (self.playery + 3) * 10]

        img_go = self.process(img_go)
        return np.transpose(mat_go, (2, 0, 1)), np.transpose(mat_lo, (2, 0, 1)), np.transpose(img_go,(2, 0, 1)), np.transpose(img_lo, (2, 0, 1))


    def process(self, img, crop=False): # copy from ProcessFrame84
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size,
                                                          resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 4])
        return x_t.astype(np.float32)
    # def identify(self, img):
    #     #start = time.process_time_ns()
    #     #img = np.array(img)
    #     print(np.array(img).shape)
    #     h, w = img.shape[0]//10, img.shape[1]//10
    #     elm_mat = np.empty((h,w,10),dtype=int)
    #     for i in range(h):
    #         for j in range(w):
    #             xs = self.axisx + np.array([i * 10] * 10)
    #             ys = self.axisy + np.array([j * 10] * 10)
    #             elm_mat[i][j]=np.sum(img[xs,ys,0:3],axis=1)
    #     elm_mat=elm_mat//3
    #     dif_mat = np.empty((h,w,len(self.elm)))
    #     for i in range(len(self.elm)):
    #         dif_mat[:,:,i]=np.sum(abs(elm_mat-self.elm[i]),axis=2)
    #     res = np.argmin(dif_mat,axis=2)
    #     for i in range(h):
    #         for j in range(w):
    #             res[i][j]=self.val[res[i][j]]
    #             if 11 <= res[i][j] <= 15:
    #                 self.playerx = i + h
    #                 self.playery = j + w
    #     res=np.lib.pad(res,((h,h),(w,w)),'constant',constant_values=0)
    #     img_extend = np.lib.pad(img,((h*10, h*10),(w*10, w*10), (0,0)),'constant',constant_values=0)
    #     mat_go=res[self.playerx-h:self.playerx+h+1 , self.playery-w:self.playery+w+1]
    #     mat_lo = res[self.playerx-2:self.playerx+3, self.playery-2:self.playery+3]
    #     mat_go=np.eye(elm_size)[mat_go]
    #     mat_lo=np.eye(elm_size)[mat_lo]
    #     img_go = img_extend[(self.playerx-h)*10: (self.playerx+h+1)*10 ,(self.playery-w)*10: (self.playery+w+1)*10]
    #     img_lo = img_extend[(self.playerx-2)*10: (self.playerx+3)*10, (self.playery-2)*10: (self.playery+3)*10]
    #     img_lo = (img_lo[:, :, 0]*0.05 + img_lo[:, :, 1] * 0.299 + img_lo[:, :, 2] * 0.537 + img_lo[:, :, 3] * 0.114)[..., np.newaxis]
    #     img_go = self.process(img_go)
    #     # out=np.argmax(mat1,axis=2)
    #     # print()
    #     # for i in range(len(out)):
    #     #     for j in range(len(out[0])):
    #     #         print(out[i][j], end=" ")
    #     #     print()
    #
    #     return np.transpose(mat_go, (2, 0, 1)), np.transpose(mat_lo, (2, 0, 1)), np.transpose(img_go, (2, 0, 1)), np.transpose(img_lo, (2, 0, 1))
    # def process(self, img, crop=True): # copy from ProcessFrame84
    #     # if img.shape == 150 * 270 * 4: # golddiger
    #     #     img = np.reshape(img, [150, 270, 4]).astype(np.float32)
    #     # elif img.shape == 90 * 130 * 4: # treasurekeeper
    #     #     img = np.reshape(img, [90, 130, 4]).astype(np.float32)
    #     # elif img.shape == 110 * 300 * 4:  # waterpuzzle
    #     #     img = np.reshape(img, [110, 300, 4]).astype(np.float32)
    #     # else:
    #     #     assert False, "Unknown resolution." + str(img.shape)
    #     img = img[:, :, 0]*0.05 + img[:, :, 1] * 0.299 + img[:, :, 2] * 0.537 + img[:, :, 3] * 0.114
    #     size = (84, 110 if crop else 84)
    #     resized_screen = np.array(Image.fromarray(img).resize(size,
    #                                                           resample=Image.BILINEAR), dtype=np.uint8)
    #     x_t = resized_screen[18:102, :] if crop else resized_screen
    #     x_t = np.reshape(x_t, [84, 84, 1])
    #     return x_t.astype(np.float32)

#buildIdentifier("golddigger",["lvl0","lvl1"]);
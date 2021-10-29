# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:56:39 2020

@author: Viony
"""
import tkinter as tk
from PIL import Image, ImageTk,ImageOps
from tkinter import filedialog
from preproses import *
from ekstrasi_fiture import *
import glob
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

class GUITBC:
    
    def __init__(self, master):
        global file_citra
        global akhir
        global in_box
        global hasil_box
        global fitur_0, fitur_45, fitur_90, fitur_135
        self.master = master
        master.title("DETEKSI TUBERCULOSIS DENGAN BOX-COUNTING DAN SVM")
        master.geometry("850x500")
        master.configure(bg='#1f6f8b')
        
        #frame 1: kanvas dan path citra
        # , bg='#a59a6f'
        self.lbXRay = tk.Label(master, text='X-Ray Thorax', anchor='w', bg='#1f6f8b', fg='white')
        self.lbXRay.place(x=50, y=20, width=150, height=20)
        self.lbXRay.configure(font=("Helvetica", 14))
        
        self.citra_test = tk.Label(master, bg='#99a8b2')
        self.citra_test.place(x=50, y=50, width=250, height=250)
        
        #frame 2: Hasil Pre-pro
        self.lbPP = tk.Label(master, text='Pre-Processing', anchor='w', bg='#1f6f8b', fg='white')
        self.lbPP.place(x=50, y=325, width=150, height=20)
        self.lbPP.configure(font=("Helvetica", 14))
        
        self.citra_pre_1 = tk.Label(master, bg='#99a8b2')
        self.citra_pre_1 = tk.Label(master, bg='#99a8b2')
        self.citra_pre_1.place(x=50, y=350, width=125, height=125)
        
        self.lbsegmentasi = tk.Label(master, text='Segmentasi', anchor='w', bg='#1f6f8b', fg='white')
        self.lbsegmentasi.place(x=200, y=325, width=150, height=20)
        self.lbsegmentasi.configure(font=("Helvetica", 14))
        self.citra_pre_2 = tk.Label(master, bg='#99a8b2')
        self.citra_pre_2.place(x=200, y=350, width=125, height=125)
                
        self.citra_pre_3 = tk.Label(master, bg='#99a8b2')
        self.citra_pre_3.place(x=350, y=350, width=125, height=125)
        
        self.lbcanny = tk.Label(master, text='Deteksi Tepi', anchor='w', bg='#1f6f8b', fg='white')
        self.lbcanny.place(x=500, y=325, width=150, height=20)
        self.lbcanny.configure(font=("Helvetica", 14))
        #self.citra_pre_4 = tk.Label(master, bg='#99a8b2' )
        #self.citra_pre_4.place(x=500, y=350, width=125, height=125)
        self.citra_pre_5 = tk.Label(master, bg='#99a8b2' )
        self.citra_pre_5.place(x=500, y=350, width=125, height=125)
                              
        
        # Button
        
        self.btOpen = tk.Button(master, text="Open Image", bd=0, command=self.fungsi_open)
        self.btOpen.place(x=350, y=50, width=120, height=32)
        self.btOpen.configure(font=("Helvetica", 12))
        
        self.btPrePro = tk.Button(master, text="Proses Citra", state=tk.DISABLED, bd=0, command=self.fungsi_proses)
        self.btPrePro.place(x=350, y=118, width=120, height=32)
        self.btPrePro.configure(font=("Helvetica", 12))
        
        self.btnBOX = tk.Button(master, text=" BOX-COUNTING ", state=tk.DISABLED, bd=0, command=self.fungsi_Box)
        self.btnBOX.place(x=350, y=186, width=120, height=32)
        self.btnBOX.configure(font=("Helvetica", 12))
        
        self.btSVM = tk.Button(master, text=" SVM ", state=tk.DISABLED, bd=0, command=self.fungsi_SVM)
        self.btSVM.place(x=350, y=254, width=120, height=32)
        self.btSVM.configure(font=("Helvetica", 12))
        
        

        
        #------------Fitur BOX-COUNTING-------------
        self.frame3 = tk.Frame(master, bg='#99a8b2')
        self.frame3.place(x=500, y=50, width=300, height=100)
        
        self.lbHBOX = tk.Label(self.frame3, text='Hasil Box-Counting', anchor='w', bg='#99a8b2', fg='white')
        self.lbHBOX.place(x=10, y=8, width=150, height=20)
        self.lbHBOX.configure(font=("Helvetica", 12))
        
        self.lbHBOX = tk.Label(self.frame3, bg='white')
        self.lbHBOX.place(x=10, y=45, width=230, height=30)
        self.lbHBOX.configure(font=("Helvetica", 12))
        
                           
       
        
        #frame 4: hasil svm
        
        self.frame4 = tk.Frame(master, bg='#99a8b2')
        self.frame4.place(x=500, y=200, width=300, height=100)
        
        self.lbSVM = tk.Label(self.frame4, text='SVM', anchor='w', bg='#99a8b2', fg='white')
        self.lbSVM.place(x=10, y=8, width=90, height=20)
        self.lbSVM.configure(font=("Helvetica", 14))
        
        
        self.hsSVM = tk.Label(self.frame4, anchor='w', bg='white')
        self.hsSVM.place(x=10, y=45, width=230, height=30)
        self.hsSVM.configure(font=("Helvetica", 14))        

        # self.label = Label(master, text="This is our first GUI!")
        # self.label.pack()

        # self.greet_button = Button(master, text="Greet", command=self.greet)
        # self.greet_button.pack()


        
    def fungsi_open(self):
        
        global file_citra
        print("tombol open di klik")
        pathh = r'CC:\Users\kntl\Documents\BelajarPC\skripsi\data_baru'
        self.nama_file =  tk.filedialog.askopenfilename(initialdir = pathh,title = "Select Image",filetypes = (("all files","*.*"),("png files","*.png")))


        self.citra_paru = Image.open(self.nama_file)
        self.citra_paru = self.citra_paru.resize((200,200), Image.ANTIALIAS)
        self.citra_paru = ImageTk.PhotoImage(self.citra_paru)

        
        self.citra_test.configure(image=self.citra_paru)
        self.citra_test.image=self.citra_paru
        self.btPrePro['state']="normal"
        # lbPath.configure(text=filename)
        # lbPath.text = filename
        print (self.nama_file)
        file_citra = self.nama_file
        return file_citra
    
    def fungsi_proses(self):
        
        global file_citra
        global in_box
        global hasil_citra
        print("tombol proses di klik")
        print ("isi : ", file_citra)
        self.citra = glob.glob(file_citra)
        self.hasil_pre, self.path, self.th3, self.mask,self.equ,self.autocanny = pre_pro(self.citra)
        # self.citra_pre_1.configure(text=isi)
        #print("Hasil otsu : ", self.th3)
        
        #hasil adaptive histogram
        self.equ = Image.fromarray(self.equ)
        self.equ = self.equ.resize((125,125), Image.ANTIALIAS)
        self.equ = ImageTk.PhotoImage(self.equ)
        self.citra_pre_1.configure(image=self.equ)
        # Hasil Otsu
        self.otsu = Image.fromarray(self.th3)
        self.otsu = self.otsu.resize((125,125), Image.ANTIALIAS)
        self.otsu = ImageTk.PhotoImage(self.otsu)
        self.citra_pre_2.configure(image=self.otsu)
        # Hasil Convex Hull
        self.hull = Image.fromarray(self.mask)
        self.hull = self.hull.resize((125,125), Image.ANTIALIAS)
        self.hull = ImageTk.PhotoImage(self.hull)
        self.citra_pre_3.configure(image=self.hull)
        # hasil deteksi tepi canny
        #self.autocanny = Image.fromarray(self.autocanny)
        #self.autocanny = self.autocanny.resize((125,125), Image.ANTIALIAS)
        #self.autocanny = ImageTk.PhotoImage(self.autocanny)
        #self.citra_pre_4.configure(image=self.autocanny)   

        # Hasil akhir
        self.akhir = Image.fromarray(self.hasil_pre)
        self.akhir = self.akhir.resize((125,125), Image.ANTIALIAS)
        self.akhir = ImageTk.PhotoImage(self.akhir)
        self.citra_pre_5.configure(image=self.akhir)
        print("Hasil akhir : ", self.akhir)
        self.btnBOX['state']="normal"
        in_box = self.path
        return in_box
        #hasil_citra = self.akhir
        #return hasil_citra
    
    
    def fungsi_Box(self):
        global in_box
        global hasil_box
        #self.hasil_citra = ImageOps.grayscale(hasil_citra)
        #img = Image.open(hasil_citra).convert('L')
        og_image = Image.open(in_box)
        # applying grayscale method
        gray_image = ImageOps.grayscale(og_image)
        #ubah ke dalam array
        img_array = np.array(gray_image)#3 jadiin array
        self.hasil_box = ekstraksi_fitur(img_array)
        self.lbHBOX.configure(text=self.hasil_box)
        self.btSVM['state']="normal"
        
       
        
    def fungsi_SVM(self):
        
        global hasil_box
        global categories
        filename = r'C:\Users\kntl\OneDrive\Documents\BelajarPC\skripsi\model_192.sav'
        model = pickle.load(open(filename, 'rb'))
        print("tombol svm di klik")
        #self.dt = pd.read_csv(r"C:\Users\kntl\Documents\Bspyder\training08042021_L_dataset.csv")
        #print(self.dt)
        data = [self.hasil_box]
        xray=data[0].reshape(1,-1)
        hasil = model.predict(xray)
        
        if hasil == 0:
            self.re = 'normal'
        if hasil == 1:
            self.re = 'TBC'
        self.hsSVM.configure(text=self.re)
        print("hasil = ", self.re)
        
root = tk.Tk()
my_gui = GUITBC(root)
root.mainloop()
from tkinter import ttk
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from keras.models import load_model
from keras import optimizers
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
#from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
#import cv2
import threading

class App:
    predictions=[]
    def __init__(self, master):
        App.count=1
        App.master=master
        #App.labels=["Spoilt", "Grade A", "Grade B", "Grade C", "Not Apple"]
        App.labels=["Spoilt", "Grade A", "Grade B", "Grade C"]
        App.labels2=["Apple", "Not Apple"]
        master.title("Apple Grading Simulator")
#root = Tk()
#root.title("A test")
        self.result=""

# Create a toolbar
        toolbar=Frame(master)
        self.b1Text="Upload Fruit"
        self.b1=Button(toolbar, text=self.b1Text, width=len(self.b1Text),command=self.upload)
        self.b1.pack(side=LEFT, padx=2, pady=2)

        self.b2Text="Grade"
        self.b2=Button(toolbar, text=self.b2Text, width=len(self.b2Text), command=self.grade, state=DISABLED)
        #self.b2.bind("<Button-1>", self.callback)
        self.b2.pack(side=LEFT, padx=2, pady=2)
        self.b3Text="Prediction Statistics"
        self.b3=Button(toolbar, text=self.b3Text, width=len(self.b3Text), command=self.plot, state=DISABLED)
        self.b3.pack(side=LEFT, padx=2, pady=2)
        toolbar.pack(side=TOP, fill=X)

    def upload(self):
        self.b3.config(state="disable")
        try:
            self.imgPath= askopenfilename(parent=App.master)
            App.master.title(self.imgPath)
            img = Image.open(self.imgPath)

            #message to tell that background subtraction is been carried out

            #img = bgSub(self.imgPath)
            #img = img.resize((100,100), Image.ANTIALIAS)
            print(self.imgPath)
            w, h=img.size
            if (w > 300 and h > 300):
                img = img.resize((500, 500), Image.ANTIALIAS)
                w,h=(500,500)
            self.img = ImageTk.PhotoImage(img)
            App.imgPath=self.imgPath
            if App.count>1:
                self.c.destroy()
                self.f.destroy()
            self.f = Frame(App.master)
            self.c = Canvas(self.f, width=w, height=h)
            self.c.create_image(0,0,anchor=NW, image=self.img)
            self.c.pack(side=LEFT, fill=BOTH)
            self.f.pack()
            self.b2.config(state="normal")
            App.count += 1
        except:
            App.master.title("Apple Grading Simulator")
            pass
        #width=224
        #height=224
        #img = img.resize((width,height), Image.ANTIALIAS)
#       img.save("resized.jpg", quality=95)

    def grade(self):
        def gradeApple():
            self.w = ttk.Progressbar(App.master, orient="horizontal", mode="indeterminate")
            self.w.pack(after=self.b2, expand=True,in_=App.master, fill=BOTH, side=TOP)
            self.w.start()
            self.b2.config(state="disable")
            model = load_model('apple_checker_model.h5')
            model2 = load_model('apple_grader_model.h5')
            model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])
            model2.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])

        #self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-5),
        #    metrics=['acc'])
            test_image = image.load_img(App.imgPath, target_size = (224, 224))
        #self.test_image = image.load_img(App.imgPath, target_size = (100, 100))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
        #self.test_image = preprocess_input(self.test_image)
            test_image /= 255.
#predict the result

            App.predictions2 = model2.predict(test_image)
            print(App.predictions2)
            y_classes2 = App.predictions2.argmax(axis=-1)
            print(y_classes2)
            if (y_classes2[0]==0):
                App.predictions = model.predict(test_image)
                print(App.predictions)
                y_classes = App.predictions.argmax(axis=-1)
                print(y_classes)
            #result=""
                if (y_classes[0]==0):
                    self.result="This Apple is spoiled"
                elif (y_classes[0])==1:
                    self.result="Grade A"
                elif (y_classes[0]==2):
                    self.result="Grade B"
                else:
                    self.result="Grade C"

            else:
                self.result="Not an apple"
            self.b3.config(state="active")
        #print(np.ndim(self.result))
            self.w.stop()
            self.w.destroy()
            top=self.top = Toplevel(App.master)
            top.title("Grading Result")
            top.geometry("130x90")
            Label(top, text=self.result).pack(pady=10)

            b = Button(top, text="OK", command=self.ok)
            b.pack(pady=5)
            self.b2.config(state="disable")
        threading.Thread(target=gradeApple).start()
    def ok(self):
        self.top.destroy()

    
    def plot(self):
        #self.model = load_model('apple_grading_model9.h5')
        #plot_model(self.model, to_file="model_plot2.pdf", show_shapes=True, show_layer_names=True)
        #top1=self.top1 = Toplevel(App.master)
        #top1.title("Prediction Statistics")
        #img = Image.open("model_plot.png")
        #w, h=img.size
        """
        self.plot = ImageTk.PhotoImage(img)
        self.f2 = Frame(top1)
        self.c2 = Canvas(self.f2, width=w, height=h)
        self.c2.create_image(0,0,anchor=NW, image=self.plot)
        self.c2.pack(side=LEFT, fill=BOTH)
        self.f2.pack(side=BOTTOM)
        """
        #predictions = self.result.argmax(axis=-1)
        if self.result == 'Not an apple':
            index = np.arange(len(App.labels2))
            predictions = []
            for i in range(2):
                predictions.append(App.predictions2[0][i])
            plt.bar(index, predictions)
            plt.xlabel("Labels", fontsize=12)
            plt.ylabel("Probabilities", fontsize=12)
            plt.xticks(index,App.labels2 , fontsize = 10, rotation = 30)
            plt.title('Predictions statistics')
            plt.show() 
        else:
            index = np.arange(len(App.labels))
            predictions = []
            for i in range(4):
                predictions.append(App.predictions[0][i])
            plt.bar(index, predictions)
            plt.xlabel("Labels", fontsize=12)
            plt.ylabel("Probabilities", fontsize=12)
            plt.xticks(index,App.labels , fontsize = 10, rotation = 30)
            plt.title('Predictions statistics')
            plt.show() 
"""def bgSub(img):
    #Load the Image
    imgo = cv2.imread(img)
    height, width = imgo.shape[:2]

    #Create a mask holder
    mask = np.zeros(imgo.shape[:2],np.uint8)

    #Grab Cut the object
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #Hard Coding the Rect… The object must lie within this rect.
    rect = (10,10,width-30,height-30)
    cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = imgo*mask[:,:,np.newaxis]

    #Get the background
    background = imgo - img1

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

    #Add the background and the image
    final = background + img1

    #To be done – Smoothening the edges….
    cv2.imwrite('newImage_without_background.jpg', final)
    image = Image.open('newImage_without_background.jpg')
    return image"""

# run the mainloop
root=Tk()
app=App(root)
root.mainloop()

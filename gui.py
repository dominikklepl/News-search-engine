from tkinter import *

import query_processor
import classifier
root = Tk()
query1 = StringVar()
date1 = StringVar()
text_classification = StringVar()

'''
def getdate():
    selection = var.get()

    if selection == 1:
        label4.visible = False
        input_date2.visible = False
        input_date1.grid(row=6, column=1)
        label3.grid(row=5)

    if selection == 2:
        label3.visible = False
        input_date1.visble = False
        label4.grid(row=5)
        input_date2.grid(row=6, column=1)
'''

def search():
    value1 = query1.get()
    value2 = date1.get()
    #print(value1)
    #print(value2)
    if value2 == "":
        #print("in if")
        value2 = None
        query_processor.search(value1, value2)
    else:
        #print("in else")
        query_processor.search(value1,value2)

def classification():
    text = text_classification.get()
    classifier.predict_topic2(text)



title = Label(root, text="Search Engine for News").grid(row=0)
label1 = Label(root, text="Search").grid(row=1)
query = Entry(root, textvariable = query1).grid(row=1, column=1)
label2 = Label(root, text="Date").grid(row=2)
label3 = Label(root, text="Format is DD/MM/YYYY or Format is DD/MM/YYYY - DD/MM/YYYY").grid(row=2, column=1)
input_date1 = Entry(root, textvariable = date1)
input_date1.grid(row=3, column=1)


# radiobutton
#var = IntVar()
#Radiobutton(root, text="Single Date", variable=var, command=getdate, value=1).grid(row=3, column=1)
#Radiobutton(root, text="Range Date", variable=var, command=getdate, value=2).grid(row=4, column=1)

button1 = Button(root, text=" Enter text for Search", fg="blue",
                 command=search).grid(row=5, column=1)

label4 = Label(root, text="Text Classification").grid(row=7)
label5 = Label(root, text="Enter Text").grid(row=8)
input_entry = Entry(root, textvariable = text_classification).grid(row=8, column=1)

button2= Button(root, text="Classification", fg="blue",
                 command=classification).grid(row=9, column=1)


root.mainloop()

import tkinter
from tkinter import messagebox
from tkinter import *


root = Tk() 
root.title("Airline Delay Predictor") 
root.geometry('600x1000')                 #是x 不是* 


l1 = Label(root, text="Aircraft Type：") 
l1.pack() 
at_text = StringVar() 
at = Entry(root, textvariable = at_text) 
at_text.set(" ") 
at.pack() 


l2 = Label(root, text="Origin Airport：") 
l2.pack()  
oa_text = StringVar() 
oa = Entry(root, textvariable = oa_text) 
oa_text.set(" ") 
oa.pack() 


l3 = Label(root, text="Destination Airport：") 
l3.pack()  
da_text = StringVar() 
da = Entry(root, textvariable = da_text) 
da_text.set(" ") 
da.pack() 




l4 = Label(root, text="Departure Weekday：") 
l4.pack()  
dw_text = StringVar() 
dw = Entry(root, textvariable = dw_text) 
dw_text.set(" ") 
dw.pack() 




l5 = Label(root, text="Departure Time：") 
l5.pack()  
dt_text = StringVar() 
dt = Entry(root, textvariable = dt_text) 
dt_text.set(" ") 
dt.pack() 

l6 = Label(root, text="Arrival Time：") 
l6.pack()  
att_text = StringVar() 
att = Entry(root, textvariable = att_text) 
att_text.set(" ") 
att.pack() 

l7 = Label(root, text="Airline：") 
l7.pack()  
a_text = StringVar() 
a = Entry(root, textvariable = a_text) 
a_text.set(" ") 
a.pack() 


l8 = Label(root, text="Aircraft Manufacturer：") 
l8.pack()  
am_text = StringVar() 
am = Entry(root, textvariable = am_text) 
am_text.set(" ") 
am.pack() 

def on_click(): 
    a1 = at_text.get()
    a2 = oa_text.get()
    a3 = da_text.get()
    a4 = dw_text.get()
    a5 = dt_text.get()
    a6 = att_text.get()
    a7 = a_text.get()
    a8 = am_text.get()
    
    
    import imp
    # Import packages and classes to use
    import warnings
    warnings.filterwarnings("ignore")
    import sys
    import itertools
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import model_selection
    from sklearn import preprocessing
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    import sklearn.tree as tree
    from sklearn.ensemble import RandomForestClassifier


    def get_hr(time):
        time=time.split(":")
        hr = int(time[0])
        return hr

    # This function takes in categorical attributes and convert them into numbers
    def type_to_num(name_list, df):
        for name in name_list:
            a_type = df[name].value_counts().keys().tolist()
            a_replace = list(range(len(a_type)))

            #print(df[name].value_counts())

            df[name].replace(a_type, a_replace, inplace = True)

        return df


    def machine_learning(array, attr , attr_names):

        # Prepare attributes array and class array
        X = array[:, attr[0 : len(attr)-1]]
        Y = array[:, attr[len(attr)-1]]


        # Splitting training and validation dataset
        X_train, X_validate, Y_train =\
        X[0:len(X)-1,:], X[len(X)-1,:],Y[0:len(Y)-1]



        # Add each algorithm and its name to the model array
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        prediction = model.predict(X_validate.reshape(1,-1))
        if prediction[0]==0:
            return('Your flight will not be delay!')
        elif prediction[0]==1:
            return('Your flight is estimated to delay less than 10 mins.')
        elif prediction[0]==2:
            return('Your flight is estimated to delay 10-30 mins.')
        elif prediction[0]==3:
            return('Your flight is estimated to delay more than 30 mins.')







    # Define main function here

    # Open cleaned file and read to a dataframe
    with open ('cleaned_data.csv', 'r') as raw:
        df = pd.read_csv(raw , sep=',', encoding='utf-8')



    attr_list = ['aircrafttype', 
                    'origin', 'destination', 'filed_departuretime_week',
                    'filed_departuretime_hr', 'estimatedarrivaltime_hr', 'airline',
                    'aircraft_manuf']
    change_list = ['aircrafttype', 'origin', 'destination',
                    'filed_departuretime_week','airline','aircraft_manuf']

    dep_delay_bin = [-3800, 0, 10, 30, 1000]

    
    df['aircrafttype'].iloc[-1]=a1
    df['origin'].iloc[-1]=a2
    df['destination'].iloc[-1]=a3
    df['filed_departuretime_week'].iloc[-1]=a4
    df['filed_departuretime_time'].iloc[-1]=a5
    df['estimatedarrivaltime_time'].iloc[-1]=a6
    df['airline'].iloc[-1]=a7
    df['aircraft_manuf'].iloc[-1]=a8



    df = df[df['filed_departuretime_time'].isna() == False]
    df = df[df['estimatedarrivaltime_time'].isna() == False]

        # Get the departure and arrival hour from time in hr:mm:ss format
    df['filed_departuretime_hr'] = df.apply(lambda x: \
        get_hr(x['filed_departuretime_time']),axis=1)



    df['estimatedarrivaltime_hr'] = df.apply(lambda x: \
        get_hr(x['estimatedarrivaltime_time']),axis=1)

    # Further cleaning before machine learning
    df = df[df['aircrafttype'].isna() == False]


    df_ml = df.loc[:,attr_list]


    df_ml = type_to_num(change_list, df_ml)


    delay_level = pd.cut(df['dep_delay_min'], \
                        dep_delay_bin, labels = range(len(dep_delay_bin)-1))

    delay_level = delay_level.tolist()

    data_arr = df_ml.values 

    norm_arr = np.c_[data_arr, delay_level]

    # Make a list containing columns index of the dataframe
    attr = list(range(len(df_ml.columns)))
    attr.append(len(df_ml.columns))

    # Call the machine learning function
    
    string = str(machine_learning(norm_arr, attr, attr_list)) 
    messagebox.showinfo(title='Delay Prediction', message = string) 
    



Button(root, text="predict your delay", command = on_click).pack() 
     
root.mainloop()
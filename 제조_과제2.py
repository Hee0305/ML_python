import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

os.getcwd()
'''
a2_load = df.loc[:,'CDJK_MCB05_A2_SPINDLE_LOAD']
a2_speed = df.loc[:,'CDJK_MCB05_A2_SPINDLE_SPEED']
'''

path = './첨단정공 5월 데이터'
df = pd.read_csv(f'{path}/CDJK 2022-05-02 00;00;00 ~ 2022-05-03 00;00;00.csv')
sl=df.loc[:,'CDJK_MCB05_A1_SPINDLE_LOAD']
sp=df.loc[:,'CDJK_MCB05_A1_SPINDLE_SPEED']
start_point=[]
peak=[]


def plot(Xlim1, Xlim2):  
    peaks, properties = find_peaks(sl, height=[40,200])
    plt.plot(sl,label='Spindle_Load')
    plt.plot(peaks, sl[peaks], 'x') # peak 지점
    plt.plot(sp,label='Spindle_Speed')
    
    plt.xlim(Xlim1, Xlim2)
    plot_line(start_point) #     
    plt.show()
    peak.append(peaks)
     
def plot_line(list_name):   
    for i in range(0,len(list_name)):
        plt.axvline(x=list_name[i],ymin=0,ymax=30,color='red', linewidth=1)

def start_points():   
    for i in range(0,863999):
        if sp[i]+500 < sp[i+1]:
            start_point.append(i+1)
 
def start_point_refine():
    start_points()
    try:
        for j in range(0,len(start_point)):
            if start_point[j]+30 > start_point[j+1]:
                del start_point[j+1]   
    except IndexError:
        print("End")
   
          
start_point_refine()             
plot(615000,625000)
plot(615300,615600)


    

    
    
    
    
    
    





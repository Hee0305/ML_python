import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 
from scipy.signal import find_peaks

path = './첨단정공 5월 데이터'
file_list=os.listdir(path)
start_point=[]
peak=[]

def start_points(speed):   
    for i in range(0,863999):
        if speed[i]+500 < speed[i+1]:
            start_point.append(i+1)
            
def start_point_refine():
    try:
        for j in range(0,len(start_point)):
            if start_point[j]+30 > start_point[j+1]:
                del start_point[j+1]   
    except IndexError:
        print("End")
        
        
def find_peak(load):  
    peaks, properties = find_peaks(load, height=[25,200]) # 20 -> 7.8852 / 25-> 7.7338 / 30 -> 6.8708 
    #peaks, properties = find_peaks(load, distance=9) # 5 -> 10.1688 / 8-> 9.3750 / 9-> 7.6396 / 10 -> 6.4596 / 15-> 5.4233
    peak.append(peaks)
    
    
def plot_line(list_name):   
    for i in range(0,len(list_name)):
        plt.axvline(x=list_name[i],ymin=0,ymax=20,color='red', linewidth=1)
        
        
def plot(load,speed):  
    peaks, properties = find_peaks(load, height=[30,200])
    plt.plot(load,label='Spindle_Load')
    plt.plot(speed,label='Spindle_Speed')
    plt.plot(peaks, load[peaks], 'x') # peak 지점
    plot_line(start_point)       
    plt.xlim(288097,288231) 
    #plt.ylim(0,200)
    plt.show()
    
peak_high = []          
for i in range(0,1): # len(file_list)
    df = pd.read_csv(f'{path}/{file_list[i]}')
    a1_load = df.loc[:,'CDJK_MCB05_A1_SPINDLE_LOAD']
    a1_speed = df.loc[:,'CDJK_MCB05_A1_SPINDLE_SPEED'] 
    start_points(a1_speed)
    start_point_refine() 
    find_peak(a1_load) 
    
    plot(a1_load,a1_speed)
    
    for j in range(1,len(start_point)): #len(start_point)
        a = start_point[j-1]
        b = start_point[j]
        load_section = a1_load[a:b] # start_point 부터 다음 start_point 까지 구간
        plt.plot(load_section)
        plt.xlim(288090,289600)
        plt.show()
        
        for k in range(0,35046): # len(peak)
            if a < peak[0][k] < b: # peak[k]가 첫번째 start_point 값과 다음 start_point 의 사이 값일 경우  
                peak_high.append(peak[0][k]) # peak[k]를 리스트에 추가한다. 
        peak_high.append()
        
    

'''
## Section plot
for k in range(1,10):
    a = start_point[k-1]
    b = start_point[k]
    load_section = df.loc[a:b,'CDJK_MCB05_A1_SPINDLE_LOAD']
    find_peak(load_section)
    plt.plot(load_section)
    plt.show() 
'''   
    
    


    

        

    
    




    

    

            

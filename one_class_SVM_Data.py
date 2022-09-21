import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import find_peaks

def interval(SPINDLE_SPEED,SPINDLE_LOAD, model):
    LOAD_DATA, ERROR_DATA, Seq_Index = [], [], []
    if model == "A1":                                                       #A1 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=500
        SPINDLE_SPEED[SPINDLE_SPEED>=400]=500
        SPINDLE_SPEED = SPINDLE_SPEED-500

    elif model == "A2":                                                     #A2 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=200]=600 # 100->200
        SPINDLE_SPEED[SPINDLE_SPEED>=600]=600
        SPINDLE_SPEED = SPINDLE_SPEED-600
        
    SPINDLE_SPEED = SPINDLE_SPEED*(-1)
    SPINDLE_SPEED[SPINDLE_SPEED>0]=300
    SPINDLE_SPEED_FF  = SPINDLE_SPEED.diff()

    for i, ff in enumerate(SPINDLE_SPEED_FF[:-200]):                        #diff 처리 후 고점, 저점 탐색
        if ff>0:
            length = SPINDLE_SPEED_FF[i:i+200]
            if min(length)<0:
                Seq_Index.append([i,length.index[length == min(length)][0]])
                    
    indexs = []                                     
    for x, y in Seq_Index:                                                  #치리된 고점, 저점을 하나의 섹터로 하고 그 섹터의 최저값을 시작 및 끝 지점으로 사이클 정의
        data = SPINDLE_LOAD[x:y]
        index = data.index[data ==min(data)]
        indexs.append(index[0])
        
    for i,j in enumerate(indexs[:-2]):                                      # 사이클에서 정상적인 범주와 비정상적인 범주 확인 후 처리
        a = j
        b = indexs[i+1]
        data = SPINDLE_LOAD[a:b]
        if (j+300 > indexs[i+1])&(data.isna().sum()==0)&((b-a)<3000)&(len(data.index[data==0])/len(data)<0.5): # 0.8-> 0.5
            LOAD_DATA.append(data)
            # print(LOAD_DATA)
            
        else:
            ERROR_DATA.append(data)
    print(len(LOAD_DATA),"\t",len(ERROR_DATA))
    # print(LOAD_DATA)
    return LOAD_DATA, ERROR_DATA

def find_peak(load):  
    peaks, properties = find_peaks(load, distance=9) # 
    peak.append(peaks)
    #print(peak)
    

total_data = []
total_error_data = []
columns = []

for path in sorted(glob.glob("첨단정공 5월 데이터/*")):
    columns.append(path.split("/")[-1].split(".")[0])
    
    print(path)
    df = pd.read_csv(path)
    SPINDLE_SPEED = df.CDJK_MCB05_A1_SPINDLE_SPEED.copy()
    SPINDLE_LOAD = df.CDJK_MCB05_A1_SPINDLE_LOAD.copy()
    LOAD_DATA, ERROR_DATA = interval(SPINDLE_SPEED,SPINDLE_LOAD, "A1")
    total_data.extend(LOAD_DATA)
    total_error_data.append(ERROR_DATA)
    # print(len(total_data))
total_error_data = pd.DataFrame([total_error_data])
total_error_data.columns = columns


# ------------------------------------------------------------------------------------------------------------
peak=[]
high_peak_list=[]
def peak_avg_var():

    pp=[]  
    for i in range(0,len(total_data)): #len(total_data)
        section = total_data[i]   
        find_peak(section)
        #plt.plot(section)
        #plt.show()
        peak_point = peak[i]
    
        for j in range(0,len(peak_point)): # 피크 평균, 분산 
            v1 = peak_point[j] # [12 23 42 131]
            v2 = section.index[v1]
            v3 = section[v2]
            pp.append(v3)
                
        high_peak_list.append(pp)
        pp=[]
         
peak_areas=[]
def peak_area():
    ll=[]
    for i in range(0,len(total_data)): #len(total_data)
        section = total_data[i]   

        for k in range(0,len(section.index)): # 면적 총 합 -> 높이가 0이 아니면 면적을 빼라 
            a1 = section.index[k]
            a2 = section[a1] # y값 == 높이 
            ll.append(a2)
        y=ll
        area = np.trapz(y) # 면적 
        ll=[]
        peak_areas.append(area)



peak_avg_var() 
peak_area()
 

###
peak_avg=[]
peak_var=[]         
for i in range(0,len(high_peak_list)):
    hp = high_peak_list[i]
    #print(f'{i}번째 : {hp}') 
    peak_average = np.around(np.mean(hp),3) # 피크 평균 
    peak_v = np.around(np.var(hp),3) # 피크 분산  
    
    peak_avg.append(peak_average)
    #print(peak_avg)

    peak_var = np.append(peak_var, peak_v)
    #print(peak_var)
###
    
    
result=[] 
result=[[peak_avg[i],peak_var[i],peak_areas[i]] for i in range(len(peak_avg))]




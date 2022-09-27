import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import find_peaks

#  Data load   -----------------------------------------------------------------------------------------------------
def interval(SPINDLE_SPEED,SPINDLE_LOAD, model):
    LOAD_DATA, ERROR_DATA, Seq_Index = [], [], []
    if model == "A1":                                                       #A1 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=500
        SPINDLE_SPEED[SPINDLE_SPEED>=400]=500
        SPINDLE_SPEED = SPINDLE_SPEED-500

    elif model == "A2":                                                     #A2 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=600
        SPINDLE_SPEED[SPINDLE_SPEED>=600]=600
        SPINDLE_SPEED = SPINDLE_SPEED-600
        
    SPINDLE_SPEED = SPINDLE_SPEED*(-1)
    SPINDLE_SPEED[SPINDLE_SPEED>0]=300
    SPINDLE_SPEED_FF  = SPINDLE_SPEED.diff()

    for i, ff in enumerate(SPINDLE_SPEED_FF[:-200]):                        #diff 처리 후 고점, 저점 탬색
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
        if (j+300 > indexs[i+1])&(data.isna().sum()==0)&((b-a)<3000)&(len(data.index[data==0])/len(data)<0.5):
            LOAD_DATA.append(data) 
        else:
            ERROR_DATA.append(data)
    print(len(LOAD_DATA),"\t",len(ERROR_DATA))

    return LOAD_DATA, ERROR_DATA



def interval_err(SPINDLE_SPEED,SPINDLE_LOAD, model):
    LOAD_DATA, ERROR_DATA, Seq_Index = [], [], []
    if model == "A1":                                                       #A1 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=500
        SPINDLE_SPEED[SPINDLE_SPEED>=400]=500
        SPINDLE_SPEED = SPINDLE_SPEED-500

    elif model == "A2":                                                     #A2 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=600
        SPINDLE_SPEED[SPINDLE_SPEED>=600]=600
        SPINDLE_SPEED = SPINDLE_SPEED-600
        
    SPINDLE_SPEED = SPINDLE_SPEED*(-1)
    SPINDLE_SPEED[SPINDLE_SPEED>0]=300
    SPINDLE_SPEED_FF  = SPINDLE_SPEED.diff()

    for i, ff in enumerate(SPINDLE_SPEED_FF[:-200]):                        #diff 처리 후 고점, 저점 탬색
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
        if data.isna().sum()==0:
            LOAD_DATA.append(data) 
        else:
            ERROR_DATA.append(data)
    print(len(LOAD_DATA),"\t",len(ERROR_DATA))

    return LOAD_DATA, ERROR_DATA


def data_preprocess(paths,m,val=False,error=False):
    # print(val,error)
    if error==False:
        interval_define = interval
    else:
        interval_define = interval_err
    total_data = []
    total_error_data = []
    columns = []
    if val ==True:
        path = paths
        columns.append(path.split("/")[-1].split(".")[0])
        
        print(path)
        df = pd.read_csv(path)
        SPINDLE_SPEED = df[f"CDJK_MCB05_{m}_SPINDLE_SPEED"].copy()
        SPINDLE_LOAD = df[f"CDJK_MCB05_{m}_SPINDLE_LOAD"].copy()
        total_data, total_error_data = interval_define(SPINDLE_SPEED,SPINDLE_LOAD, m)
    else: 
        for path in sorted(paths):
            columns.append(path.split("/")[-1].split(".")[0])
            
            print(path)
            df = pd.read_csv(path)
            SPINDLE_SPEED = df[f"CDJK_MCB05_{m}_SPINDLE_SPEED"].copy()
            SPINDLE_LOAD = df[f"CDJK_MCB05_{m}_SPINDLE_LOAD"].copy()
            LOAD_DATA, ERROR_DATA = interval_define(SPINDLE_SPEED,SPINDLE_LOAD, m)

            total_data.extend(LOAD_DATA)
            total_error_data.append(ERROR_DATA)
        total_error_data = pd.DataFrame([total_error_data])
        total_error_data.columns = columns

    return total_data, total_error_data

paths = sorted(glob.glob("첨단정공 5월 데이터_복사본/*")) # paths = sorted(glob.glob("첨단정공 5월 데이터_복사본/*"))
train_path = paths[:19]
val_path = paths[19]    
test_path = paths[20:]
#print("train")
train_data, train_error_data = data_preprocess(train_path,"A1")
#print("val")
val_data, val_error_data = data_preprocess(val_path,"A1",val=True)
#print("test")
test_data, test_error_data = data_preprocess(test_path,"A1")



# Data pre-processing------------------------------------------------------------------------------------------------------------

def peak(data):
    peak=[]
    high_peak_list=[]  
    pp=[]

    for i in range(0,len(data)): #len(total_data)             
        section = data[i]   
        peaks, properties = find_peaks(section, distance=9) # 
        peak.append(peaks)  
        peak_point = peak[i]
    
        for j in range(0,len(peak_point)): # 피크 평균, 분산 
            v1 = peak_point[j] # [12 23 42 131]
            v2 = section.index[v1]
            v3 = section[v2]
    
            pp.append(v3)
           
        high_peak_list.append(pp)

    return high_peak_list

train_peak = peak(train_data)
test_peak = peak(test_data)
val_peak = peak(val_data)



def peak_area(data): 
    peak_areas=[]
    
    ll=[]
    for i in range(0,len(data)): # data = train_data, test_data, val_data
        section = data[i]   

        for k in range(0,len(section.index)): # 면적 총 합 -> 높이가 0이 아니면 면적을 빼라 
            a1 = section.index[k]
            a2 = section[a1] # y값 == 높이 
            ll.append(a2)
        y=ll
        area = np.trapz(y) # 면적 
        ll=[]
        peak_areas.append(area)
        
    return peak_areas
  
train_peak_areas = peak_area(train_data)
test_peak_areas = peak_area(test_data)
val_peak_areas = peak_area(val_data)
        
   
def peak_avg(data):
    peak_avg=[]
     
    for i in range(0,len(data)):
        hp = data[i]
        peak_average = np.around(np.mean(hp),3) # 피크 평균  
        peak_avg.append(peak_average) # 평균 리스트에 추가
        
    return peak_avg

train_avg = peak_avg(train_data)
test_avg = peak_avg(test_data)
val_avg = peak_avg(val_data)
print('Avg 끝')
def peak_var(data):
    peak_var=[]   
    
    for i in range(0,len(data)):
        hp = data[i]
        peak_v = np.around(np.var(hp),3) # 피크 분산  
        peak_var = np.append(peak_var, peak_v) # 분산 리스트에 추가 
        
    return peak_var

train_var = peak_var(train_data)
test_var = peak_var(test_data)
val_var = peak_var(val_data)
print('Var 끝')


    
result_train=[] 
result_train=[[train_avg[i],train_var[i],train_peak_areas[i]] for i in range(len(train_avg))]

result_test=[] 
result_test=[[test_avg[i],test_var[i],test_peak_areas[i]] for i in range(len(test_avg))]

result_val=[] 
result_val=[[val_avg[i],val_var[i],val_peak_areas[i]] for i in range(len(val_avg))]

def count(list_name): ## 오답, 정답 count
    wrong_num=0
    right_num=0
    for i in range(0,len(list_name)):
        if list_name[i]==-1:
            wrong_num = wrong_num+1
        elif list_name[i]==1:
            right_num = right_num+1
    print(f'오답 : {wrong_num}개, 정답 : {right_num}개')
    print(f'Score 평균 : {test_score.mean()}')


#------------------------------------------------------------------------------------------------------------------

from sklearn.svm import OneClassSVM

train = result_train
val = result_val
test = result_test


clf = OneClassSVM(gamma=0.001, nu=0.03, kernel='rbf').fit(train) 

# val_pred= clf.predict(val)
# val_score = clf.score_samples(val)

test_pred = clf.predict(test)
test_score = clf.score_samples(test)

count(test_pred)

#------------------------------------------------------------------------------------------------------------------

pos = np.where(test_pred==-1)

e_list=[]

for i in range(0,len(pos[0])):
    e_list.append(test_data[pos[0][i]].index.start)


to_taeyoung = pd.DataFrame(e_list, columns=['index'])

to_taeyoung.to_csv('C:/Users/Hee/Desktop/Python/to_taeyoung_A1.csv')
    



    

    







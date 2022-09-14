import pandas as pd
import os

df = pd.read_csv('./첨단정공 5월 데이터/CDJK 2022-05-02 00;00;00 ~ 2022-05-03 00;00;00.csv')
bad_count = df.loc[:,'CDJK_MCB05_A_BAD_COUNT']
bc_list = list(bad_count)

li=[]
index_list=[]
index=[]
column=[]
def find_inflection_point():
    for i in range(0,863999):
        if bad_count[i] < bad_count[i+1]:
            li.append(i)
        

def find_index():   
    for j in range(0,len(li)):
        last=len(li)-1
        
        if li[j]!=li[last] and li[j]+200<li[j+1]: # 만약 불러온 값이 마지막값이 아니고, 불러온 값+200 이 다음 인덱스값보다 작을때
            index_list.append(bad_count[li[j]-200:li[j]]) # bc_list에서 불러온값의 200개 전부터 불러온 값 까지를 추가해라 
            
            for i in range(0, len(index_list)):             
                idx = index_list[i].index[:] # 인덱스 추출
                idx2 = list(idx) # 리스트로 변환 
            index.append(idx2) # index 리스트에 추가 
                
                
            
        elif li[j] == li[len(li)-1]: # 만약 변곡점 리스트의 마지막 값 이라면
            index_list.append(bad_count[li[last]-200:li[last]]) # 마지막값의 200개 전부터 마지막 값 까지를 추가해라  
            
            for i in range(0, len(index_list)):             
                idx = index_list[i].index[:] 
                idx2 = list(idx)  
            index.append(idx2) 
        

def column():
    print(len(index))
    for i in range(0,len(index)):
        cc = [1 for i in range(200)]
        column.append(cc)
        
find_inflection_point()       
find_index()

dd=[]

for i in range(0, len(index)):
    first_val = index[i][0] - 1
    front200 = list(range(first_val-199, first_val+1))
    print(front200)
    print(len(front200))
    
    if front200 not in li and front200 not in index[1]:
        dd.append(front200)
        
dd.clear()      
    

a=6
b=[1,2,3,4,5,6]

if a in b:
    print('있다')
else:
    print('없다')

    
    
    
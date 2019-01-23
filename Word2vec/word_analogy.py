import os
import pickle
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))


path='.\word_analogy_dev.txt'
dev_exp = open(path,'r')
dev_file=dev_exp.readlines()
dev_file[len(dev_file)-1]=dev_file[len(dev_file)-1]+'\n' 
pipe=[]
for i in dev_file:
	pipe.append(i.split("||")) 

#Splitting by pipe
first_half=[]
second_half=[]
for i in range(len(pipe)):
	first_half.append(pipe[i][0].split(",")) 
	second_half.append(pipe[i][1].split(","))


avg_diff_first_half = []
diff_second_half = []
similarity = []
flag_1=0
flag_2=0
temp_flag=1

for i in range(len(pipe)):
    
	#Dividing half before pipe and calculating difference vector
    temp_average = 0 
    for group in first_half[i]:
        k = group[1:len(group)-1]
        l = k.split(':')
        difference = embeddings[dictionary[l[0].replace("\"","")]] - embeddings[dictionary[l[1].replace("\"","")]] 
        temp_average = temp_average + difference
        
    temp_average = temp_average/3 
    avg_diff_first_half.append(temp_average)
	
    #Dividing half after pipe and calculating difference vector
    temp = []
    for j in range(len(second_half[i])):
        if j == len(second_half[i])-1:
            k=second_half[i][j][1:len(second_half[i][j])-2]
        else:
            k=second_half[i][j][1:len(second_half[i][j])-1]
        l=k.split(':')
        difference = embeddings[dictionary[l[0]]] - embeddings[dictionary[l[1]]] 
        temp.append(difference)
        
    diff_second_half.append(temp)
    
    #Calculating cosine similarity
    temp1 = []
    for diff_val in diff_second_half[i]:
        c = 1-spatial.distance.cosine(diff_val,avg_diff_first_half[i]) 
        temp1.append(c)
    similarity.append(temp1)
    

#for i in range(0,len(similarity[:300]),200):
#   print(similarity[i])
#print(diff_second_half[1][1])
flag=0
out_file = './new_2.txt' 
out_fdr = open(out_file,'w')

for i in range(len(pipe)):
   min_index=0
   max_index=0
   for j in range(len(similarity[i])):
      
      if  similarity[i][max_index] < similarity[i][j]: 
         max_index = j

      if similarity[i][min_index] > similarity[i][j] : 
         min_index = j
   str_out = ""
   for pair in (second_half[i]):  
     str_out = str_out + pair + " "
   str_out=str_out[0:len(str_out)-2] + " "
   str_out = str_out + second_half[i][min_index] + " "
   #print(str_out)
   str_out = str_out + second_half[i][max_index]
   str_out=str_out.replace("\n","")
   out_fdr.write(str_out+"\n")

out_fdr.close()
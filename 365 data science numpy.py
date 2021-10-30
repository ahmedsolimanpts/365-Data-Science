#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


array_a = np.array([[1,2,3],[4,5,6]],dtype=np.int16)
array_a


# In[3]:


array_b = np.array([[1],[1]])


# In[4]:


a = np.multiply(array_b,array_a,dtype=np.float32)


# In[5]:


a[1] # 1-D Array


# In[6]:


a[1:] # 2- D A rray


# ### Matrix[a:b:c,a:b:c] ----- > [:::,:::] a -> Start , b-> END , c -> Step

# In[7]:


type(array_a[:,1])


# In[8]:


array_a[:,1]


# In[9]:


np.squeeze(array_a[0,1:])


# In[10]:


type(np.squeeze(array_a[0,1:]))


# In[11]:


empty =np.empty(shape=(2,3),dtype=np.int8)
empty


# In[12]:


zeros = np.zeros(shape=(3,3),dtype=np.int16)
zeros


# In[13]:


ones = np.ones(shape=(3,3),dtype=np.int16)
ones


# In[14]:


fill = np.full(shape=(3,3),fill_value=2) # fill_value is requierd 
fill


# #### np_like

# #### np _like is used with np.empty , zeros , ones , full Function

# In[15]:


array_a


# In[16]:


array_a_like =np.zeros_like(array_a)
array_a_like


# #### np_arange(stop,start,step)

# In[17]:


array_arange =np.arange(30)
array_arange


# In[18]:


from numpy.random import Generator as gen


# In[19]:


from numpy.random import PCG64 as pcg


# In[20]:


array_ran =gen(pcg()) # seed =


# In[21]:


array_ran.normal(size=(3,3))


# In[22]:


array_int = gen(pcg(seed=365))
array_int.integers(low=10,high=20,size=(4,4))


# In[23]:


array_int = gen(pcg(seed=365))
array_int.random(size=(4,4))


# In[24]:


array_int = gen(pcg())
array_int.choice([1,2,3,4,5],p=[.1,.1,.1,.1,.6],size=(4,4)) # p -> Propapility of each elemnt to show  all equal ! , optinal


# In[25]:


array_int = gen(pcg())
array_int.poisson(lam = 10, size=(4,4)) # lam -> lambda


# In[26]:


array_int = gen(pcg())
array_int.binomial(n=100,p=0.4,size=(4,4)) 


# In[27]:


array_int = gen(pcg())
array_int.logistic(loc=9,scale=1.2,size=(4,4)) # loc -> location 


# # Useful Function

# #### .transpose() -> this function used to transpose matrix col -> row 

# ### This function use for save array in new file in the directory

# <p> np.savetxt('file_name.extions',delimiter=' , ',fmt='%s',array_data)<br>
#     np.save('file_name',array_data) <b>-> this save data in .npy extinons and dont convert array format</b> <br>
#     np.savez('file_name',label_data = array_data) <b>-> this save data in .npz extinons and dont convert array format and can store more than one array </b> <br>
#     np.genfromtxt('file_name.extions',delimiter =' , ') <br>
#     There is also np.loadtxt('file_name.extions',delimiter =' , ')
#     <br>
#     <h2>genfromtxt & loadtxt </h2><br>
#     this two function do same thing but<br>
#     genfromtxt can handle miss data and loadtxt is faster than genfromtxt but cant handle miss data
#     <br>
#     <h4>paramters of genfromtxt </h4><br>
#     <ol>
#     <li>Skip_header = </li> 
#     <li>Skip_footer = </li>
#     <li>usecols(i,j) -> <b>this return only columns i , j from the file</b></li> 
#     <li>unpack = True --><b> this split the array columns to more one variable</b> </li>
#     </ol>
# <br>
# <br>
# 
# np.array_equal(array_1,array_2) -> This function check if the two array are the same and return boolean
# 
# 
# 
# </p>

# In[28]:


array_a


# In[29]:


print('mean: ',np.mean(array_a))
print('min : ',np.min(array_a))
print('max : ',np.max(array_a))
print(np.min(array_a,axis=0)) # axis = 0 -> min in the col 1 -> in the row
print('ptp : ',np.ptp(array_a)) # the range max - min
print('percentile : ',np.percentile(array_a,70)) # -> the 70% of values up tp 5
print(np.minimum(array_a[0],array_a[1]))   #-> this function take two array and get all the min in the same place and return a new array
print('median : ',np.median(array_a))
print('var : ',np.var(array_a))
print('std :',np.std(array_a))


# In[30]:


print('covriance :')
print(np.cov(array_a))


# In[31]:


print('correlation :')
print(np.corrcoef(array_a))


# In[32]:


print(np.histogram(array_a)[0]) #-> selice the first array of the output 
np.histogram(array_a) # there is bins , range to set your values


# In[33]:


import matplotlib.pyplot as plt
plt.hist(array_a.flat)
plt.show()


# np.nanmean ,np.nanmedian, np.nanvar ..etc this function are used when we have nan data in array and we want min, max ,mean etc without fill nan with corrct data 

# # Missing data 

# In[34]:


# np.isnan().sum() -> this function sum all nan Values in the matrix


# <p>
#     d_mean =np.nanmean(array,axis=1)<b> -->this get all mean for each column and stor it as array</b><br>
#     -----------------------------------------------------------------------------------------------<br>
#     data[:,col] = np.where(data_arr[:,col] ==d_mean[col],<b>  -> this function filter all row in col that equal the mean</b><br>
#     d_mean[col],<br>
#     data_arr[:,col]
#     )
#        </p>  

# <p>
#     d_mean =np.nanmean(array,axis=1)<b> -->this get all mean for each column and stor it as array</b><br>
#     ---------------------------------------------------------------<br>
#     for i in range(data.shape(1)):<b> -> shape 1 return number of columns</b> <br>
#     data[:,i] = np.where(data_arr[:,i] ==d_mean[i],<b>  -> this function filter all row in col that equal the mean</b><br>
#     d_mean[i], <b>  HERE WE ASSIGN NEW DATA  </b> <br>
#     data_arr[:,i]
#     )
#        </p>  

# <p>
#         ---------------------------------------------------------------<br>
#     <h3>HERE WE PUT ALL NEGATIVE VALUE TO ZERO</h3>
#     ---------------------------------------------------------------<br>
#     for i in range(data.shape(1)):<b> -> shape 1 return number of columns</b> <br>
#     data[:,i] = np.where(data_arr[:,i] < 0 <b>  -> this function filter all row in col that equal the mean</b><br>
#     0,<b>HERE WE ASSIGN NEW DATA</b><br> 
#     data_arr[:,i]
#     )<br>
#     ----------------------------------------------------------------------------------------------------------
#        </p>  

# In[35]:


array_a


# ##### reshape is put elemnt in order of the old matrix in the new matrix but in defferent size of col & row

# In[36]:


np.reshape(array_a,(3,2)) # (3,2) -> is a new matrix


# In[37]:


np.transpose(array_a)


# In[38]:


np.delete(array_a,0,axis=1) # 0 -> column 0


# In[39]:


np.delete(array_a,0,axis=0) # 0 -> row 0


# In[40]:


np.sort(array_a)


# In[41]:


np.sort(-array_a)


# In[42]:


-np.sort(-array_a)


# In[43]:


array_a[0:1].sort()
array_a


# In[44]:


np.sort(array_a,axis=None)


# In[45]:


np.argsort(array_a) #return the position when the matrix is sorted


# In[46]:


np.argsort(array_a ,axis=0) #return the position when the matrix is sorted


# In[47]:


np.argwhere(array_a)


# In[48]:


np.argwhere(array_a >3) # get the index col,row for elemnt greater thean 3


# In[49]:


np.argwhere(array_a %2 == 0) # get the index col,row for elemnt that it is even


# In[50]:


np.argwhere(np.isnan(array_a))


# #### get The index of the all nan in the matrix and replace them with new value
# <b>for array_index in np.argwhere(np.isnan(array_a)):<br>
#     array_a[array_index[0],array_index[1]]= Value</b>

# data =np.loadtxt('',)[:8] -> this get only 8 row of the data

# In[51]:


array_a


# # Shuffle 
# <b>is replace the row in random</b>

# In[52]:


np.random.shuffle(array_a)
array_a


# In[53]:


Matrix_A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
Matrix_A


# In[54]:


np.random.shuffle(Matrix_A)
Matrix_A


# # Casting
# <br>
# <b>
# is convert from type to another type <br>
# to convert from str to int you must convert str -> float -> int <br>
# you can convert float -> int
# </b>

# In[55]:


Matrix_A = Matrix_A.astype(dtype=np.str)


# In[56]:


Matrix_A


# In[57]:


Matrix_A.astype(dtype=np.int16)


# # Strip 
# <b>is delete text from char using np.chararray.strip() Function</b>

# ### data = np.chararray.strip(array_we_need,'char_we_need_to_delete')
# ### data = np.chararray.strip(array_we_need[:,0],'char_we_need_to_delete') -> applay on all row in columns zero
# 

# ------------------------------------------------------------------------------------------------------------------------ <br>
# # Stack
# #### stack is used to combine an array in top of another array like concatenation but must two array are the same size
# #### there are stack, hstack,vstack,dstack
# ## np.stack((array_1,array_2),axis=0)
# <h3>
# <ul>
#     <li>hstack ---> horizontal STACK</li>
#     <li>vstack ---> Vertical STACK</li>
#    <li>dstack ---> Dymnsion STACK</li>
#     
# </ul>
# </h3>

# # concateinate
# #### is combine two array with different size or same size in new array vertically or herzontaily or D-array Depond on axis number
# ### np.concatenate((array_1,array_2),axis=)
# #### axis value [0,1,2]

# # Unique 
# #### return all unique value in matrix and sorted 
# ### np.unique(array,return_counts=,return_index=) 
# #### return counts -> return # of each unique value repeat

# In[ ]:





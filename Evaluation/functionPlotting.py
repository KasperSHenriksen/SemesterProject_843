import matplotlib.pyplot as plt
import numpy as np

#Function to plot
def relu(t):
    temp = []
    for value in t:
        temp.append(np.maximum(0,value))
    return temp

def leakyrelu(t):
    temp = []
    for value in t:
        if value >0:
            temp.append(np.maximum(0,value))
        else:
            temp.append(value*0.1)
    return temp


def sigmoid(t):
    temp = []
    for value in t:
        temp.append(1/(1+np.exp(-value)))
    return temp


t1 = np.linspace(-20,20,num=2000)
#plt.subplot(211)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')


# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_ylim(0,20)



plt.plot(t1,leakyrelu(t1))
plt.show()

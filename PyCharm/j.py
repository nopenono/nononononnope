import tflearn
import numpy as np

#from tflearn.datasets import


from tflearn.data_utils import load_csv
data, labels = load_csv('ww2.csv',target_column=5,  categorical_labels=True, n_classes=2)

import csv
#.download_dataset('ww2.csv')

with open('ww2.csv','r') as f:
    reader = csv.reader(f)
    # data = list(reader)


for i in data:
    if i [0]:
        if i[0] == 'soviet':
            i[0] = 0
        elif i[0] == 'germany':
            i[0] = 1
        elif i[0] == 'poland':
            i[0] = 2
        elif i[0] == 'netherlands':
            i[0] = 3
        elif i[0] == 'china':
            i[0] = 4
        elif i[0] == 'france':
            i[0] = 5
        elif i[0] == 'gb':
            i[0] = 6
        elif i[0] == 'finland':
            i[0] = 7
        elif i[0] == 'italy':
            i[0] = 8
        elif i[0] == 'japan':
            i[0] = 9
        else:
            i[0] = 10
    if i [1]:
        if i[1] == 'air':
            i[1] = 0
        elif i[1] == 'sea':
            i[1] = 1
        else:
            i[1] = 2
    if i[2]:
        if i[2] == 'heavy':
            i[2] = 0
        elif i[2] == 'light':
            i[2] = 1
        elif i[2] == 'medium':
            i[2] = 2
        elif i[2] == 'artillery':
            i[2] = 3
        elif i[2] == 'total':
            i[2] = 4
        elif i[2] == 'totalair':
            i[2] = 5
        elif i[2] == 'armoredCars':
            i[2] = 6
        elif i[2] == 'battleships':
            i[2] = 7
        elif i[2] == 'carriers':
            i[2] = 8
        elif i[2] == 'destroyers':
            i[2] = 9
        elif i[2] == 'submarines':
            i[2] = 10
        elif i[2] == 'frigates':
            i[2] = 11
        elif i[2] == 'cruisers':
            i[2] = 12
        elif i[2] == 'bomber':
            i[2] = 13
        elif i[2] == 'fighter':
            i[2] = 14
        elif i[2] == 'CAS':
            i[2] = 15
        else:
            i[2] = 16
    if i[3]:
        i[3] = int(i[3])
    if i[4]:
        i[4] = int(i[4])
    # if i[5]:
    #     i[5] = int(i[5])




print('nation,class,type, casualties, produced, outcome')
print(data)


net = tflearn.input_data(shape=[None, 5]) #An input layer, with variable input size of examples with 6 features (the [None, 6])
net = tflearn.fully_connected(net, 32) #Two hidden layers with 32 nodes
net = tflearn.fully_connected(net, 32) #net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 32) #net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 32) #net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 2, activation='softmax') #An output later of 2 nodes, and a "softmax" activation (more on activations later)
net = tflearn.regression(net) #find the pattern





# SSoviet union = 0, Germany = 1, Poland = 2, Netherlands = 3, China = 4, France = 5, British Empire = 6, Finland = 7, Italy = 8, Empire of Japan = 9, United States = 10
model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=100, batch_size=16, show_metric=True)
# # nation,class,type, casualties, produced, outcome'
print("Jake", model.predict([[5, 0, 14, 30045, 38786]])[0][0]) # chance of loosing
print("Jake", model.predict([[5, 0, 14, 30045, 38786]])[0][1]) # chance of winning
# #
# #















import random
import math

class Neuron:


    # activation_f = 'linear' or 'relu'
    # weights : array of weights for neurons of next layer ; None if output layer
    # bias : bias for neuron
    # input_signal : if input neuron then input, else, sum of previous layer neurons (weighted)
    # output_signal : output of neuron (post activation)
    # gradient : the gradient (None if not computed)
    

    def __init__(self, activation_f, weights, bias):

        self.weights=weights
        self.bias=bias
        self.activation_f=activation_f

        self.input_signal=None
        self.output_signal=None
        self.gradient=None
    def activate(self,s):
        #take a sum, return the activation
        if self.activation_f=='linear':
            return s
        elif self.activation_f=='relu':
            return max(0,s)
        elif self.activation_f=='sigmoid':
            raise Exception("Not implemented")
            try:
                return 1/(1+math.exp(-s))
            except OverflowError:
                return 0
    def get_der(self):
        #return the derivative of the activation function
        if self.activation_f=='linear':
            return 1
        elif self.activation_f=='relu':
            if self.input_signal>0:
                return 1
            else:
                return 0
        # elif self.activation_f=='sigmoid':
        #         return self.input_signal*(1-self.input_signal)
          
INPUT_LAYER_SIZE=2
MIDDLE_LAYER_SIZE=4
MIDDLE2_LAYER_SIZE=4
OUTPUT_LAYER_SIZE=2

def init_weight_random():
    return random.randint(-10,10)/13

input_layer=[]
middle_layer=[]
middle2_layer=[]
output_layer=[]

for i in range(INPUT_LAYER_SIZE):
    input_layer.append(Neuron('linear', [init_weight_random() for j in range(MIDDLE_LAYER_SIZE)], 0))

for i in range(MIDDLE_LAYER_SIZE):
    middle_layer.append(Neuron('relu', [init_weight_random() for j in range(MIDDLE2_LAYER_SIZE)], 0))
for i in range(MIDDLE2_LAYER_SIZE):
    middle2_layer.append(Neuron('relu', [init_weight_random() for j in range(OUTPUT_LAYER_SIZE)], 0))

for i in range(OUTPUT_LAYER_SIZE):
    output_layer.append(Neuron('linear', None, 0))

# input_layer.append(Neuron('linear', [0.1,0.2,0.3], 0))
# input_layer.append(Neuron('linear', [0.4,0.5,0.6], 0))

# middle_layer.append(Neuron('relu', [1,0], 0))
# middle_layer.append(Neuron('relu', [1,0], 0))
# middle_layer.append(Neuron('relu', [0,0], 0))

# output_layer.append(Neuron('linear', None, 0))
# output_layer.append(Neuron('linear', None, 0))



def forward_propagation(input_data):
    for i in range(INPUT_LAYER_SIZE):
        input_layer[i].input_signal=input_data[i]
        input_layer[i].output_signal=input_data[i]+input_layer[i].bias

    for i in range(MIDDLE_LAYER_SIZE):
        sum=0
        for j in range(INPUT_LAYER_SIZE):
            sum+=input_layer[j].output_signal*input_layer[j].weights[i]
        middle_layer[i].input_signal=sum
        sum=middle_layer[i].activate(sum)
        sum+=middle_layer[i].bias
        middle_layer[i].output_signal=sum

    for i in range(MIDDLE2_LAYER_SIZE):
        sum=0
        for j in range(MIDDLE_LAYER_SIZE):
            sum+=middle_layer[j].output_signal*middle_layer[j].weights[i]
        middle2_layer[i].input_signal=sum
        sum=middle2_layer[i].activate(sum)
        sum+=middle2_layer[i].bias
        middle2_layer[i].output_signal=sum
        

    for i in range(OUTPUT_LAYER_SIZE):
        sum=0
        for j in range(MIDDLE2_LAYER_SIZE):
            sum+=middle2_layer[j].output_signal*middle2_layer[j].weights[i]
        output_layer[i].input_signal=sum
        sum=output_layer[i].activate(sum)
        sum+=output_layer[i].bias
        output_layer[i].output_signal=sum
    
    return [output_layer[i].output_signal for i in range(OUTPUT_LAYER_SIZE)]




def compute_gradients(ground_truth,network_output,print_verbose=False):
    #Here we define the loss function as the sum of the squared errors for each output neuron

    #the gradient is the partial derivative of the loss function with respect to the output of the neuron times the total loss

    for i in range(OUTPUT_LAYER_SIZE): 
        output_layer[i].gradient=2*(network_output[i]-ground_truth[i])
    if(print_verbose):
        print(f"output_layer[0].gradient={output_layer[0].gradient}")
        print(f"output_layer[1].gradient={output_layer[1].gradient}")
        print('===========================')
    for i in range(MIDDLE2_LAYER_SIZE):
        sum=0
        for j in range(OUTPUT_LAYER_SIZE):
            sum+=output_layer[j].gradient*middle2_layer[i].weights[j]*output_layer[j].get_der()
        middle2_layer[i].gradient=sum
    if(print_verbose):
        print(f"middle2_layer[0].gradient={middle_layer[0].gradient}")
        print(f"middle2_layer[1].gradient={middle_layer[1].gradient}")
        print(f"middle2_layer[2].gradient={middle_layer[2].gradient}")
        print('===========================')

    for i in range(MIDDLE_LAYER_SIZE):
        sum=0
        for j in range(MIDDLE2_LAYER_SIZE):
            sum+=middle2_layer[j].gradient*middle_layer[i].weights[j]*middle2_layer[j].get_der()
        middle_layer[i].gradient=sum
    if(print_verbose):
        print(f"middle_layer[0].gradient={middle_layer[0].gradient}")
        print(f"middle_layer[1].gradient={middle_layer[1].gradient}")
        print(f"middle_layer[2].gradient={middle_layer[2].gradient}")
        print('===========================')



    for i in range(INPUT_LAYER_SIZE):
        sum=0
        for j in range(MIDDLE_LAYER_SIZE):
            sum+=middle_layer[j].gradient*input_layer[i].weights[j]*middle_layer[j].get_der()
        input_layer[i].gradient=sum
    if(print_verbose):
        print(f"input_layer[0].gradient={input_layer[0].gradient}")
        print(f"input_layer[1].gradient={input_layer[1].gradient}")


def update_step(lr):
    for i in range(OUTPUT_LAYER_SIZE):

        #update bias
        output_layer[i].bias-=lr*output_layer[i].gradient
    
    for i in range(MIDDLE2_LAYER_SIZE):
        middle2_layer[i].bias-=lr*middle2_layer[i].gradient
        for j in range(OUTPUT_LAYER_SIZE):
            middle2_layer[i].weights[j]-=lr*output_layer[j].gradient*middle2_layer[i].output_signal

    for i in range(MIDDLE_LAYER_SIZE):
        middle_layer[i].bias-=lr*middle_layer[i].gradient
        for j in range(MIDDLE2_LAYER_SIZE):
            middle_layer[i].weights[j]-=lr*middle2_layer[j].gradient*middle_layer[i].output_signal #TODO? should there be another derivitive here

    for i in range(INPUT_LAYER_SIZE):
        input_layer[i].bias-=lr*input_layer[i].gradient
        for j in range(MIDDLE_LAYER_SIZE):
            input_layer[i].weights[j]-=lr*middle_layer[j].gradient*input_layer[i].output_signal


#make goal to be XOR function
#input will be 0 0, 0 1, 1 0 or 1 1
losses=[]

for k in range(500):
    sum_loss=0
    for i in range(4):
            input_data=[i//2,i%2]
            network_output=forward_propagation(input_data)
            compute_gradients([1,1] if i==1 else [0,0],network_output,print_verbose=False)
            sum_loss+=abs(output_layer[0].gradient)+abs(output_layer[1].gradient)
            #print(f"network_output={network_output} for input {input_data}")
            
            
            update_step(0.03)
    print(f"sum loss is {sum_loss}")

    losses.append(sum_loss)

#plot with matplot lib losses over time

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()






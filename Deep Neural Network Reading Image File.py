def create_matrix(dim1, dim2):
    """
    Additional Function made:

    Function used to convert lists into matrix (nested lists)

    Input: 2 Integers, for the dimensions
    Output: a matrix (nested lists, with elements 0) e.g [[0, 0], [0, 0], [0, 0]]

    Usage Examples
    Input dimensions such as 3 and 4, would output [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    Made this additional function to quickly convert a single list into nested list, as there were times in which given output had to be converted into matrix.
    Made an empty list called “Matrix”, and made rows as it iterated through i, and made columns as it iterated through j (nested the for loops). Appended the elements of the lists by giving it values 0.

    """

    matrix = []
    
    for i in range(dim1):

        matrix.append([])
        
        for j in range(dim2):

            matrix[i].append(0)                        

    return(matrix)


def linear(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a list of weights (w) and a bias (b).
    Output: A single number corresponding to the value of f(x) in Equation 1.

    >>> x = [1.0, 3.5]
    >>> w = [3.8, 1.5]
    >>> b = -1.7
    >>> round(linear(x, w, b),6) #linear(x, w, b)

    Function that outputs an individual vertex in the output layer.
    
    This is done by multiplying the same index position of the different lists, and finding the sum.

    <expalined as commments next to code>

    7.35
    """
    pass
    output = 0

    for i in range(0, len(w)):

        output += (x[i]*w[i]) 

    # multiplies each value in the same position and assigns it to a variable output

    output += b

    # each value is added to b

    return output


def linear_layer(x, w, b): # 1 Mark
    """
    Output layer

    Input: A list of inputs (x), a table of weights (w) and a list of 
           biases (b).
    Output: A list of numbers corresponding to the values of f(x) in
            Equation 2.
    
    >>> x = [1.0, 3.5]
    >>> w = [[3.8, 1.5], [-1.2, 1.1]]
    >>> b = [-1.7, 2.5]
    >>> y = linear_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #linear_layer(x, w, b)
    [7.35, 5.15]

    This function returns the output of the whole layer

    <expalined as commments next to code>


    """
    pass

    result_list = [] 

    # made an empty list to store the output values

    for i in range(0, len(b)):

        # made it the length of bias because it is the number you add to the output = number of repetition have to much the number of times of iteration

        output = 0

        output += b[i]

        for j in range(0, len(x)):

            output += (x[j]*w[i][j])

        # iterates through the positions of the lists, and multiplies x and w
        result_list += [output]
        


    return result_list



def inner_layer(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a 
           list of biases (b).
    Output: A list of numbers corresponding to the values of f(x) in 
            Equation 4.

    >>> x = [1, 0]
    >>> w = [[2.1, -3.1], [-0.7, 4.1]]
    >>> b = [-1.1, 4.2]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [1.0, 3.5]
    >>> x = [0, 1]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [0.0, 8.3]

    Function that outputs the vertices in the inner layers.
    
    <expalined as commments next to code>

    """
    pass

    result = create_matrix(len(x), len(w[0])) # uses the additionally created function, and gives the input of the dimensions

    answer = []

    result_list = [] 

    for i in range(0, len(b)):
        output = 0
        output += b[i]

        # adds the bias through each iteration

        for j in range(0, len(x)):
            output += (x[j]*w[i][j])

        # output is calcuated by matrix multilplication
        
        result_list += [max(output,0)] 
        


    return result_list


def inference(x, w, b): # 2 Marks
    """
    Input: A list of inputs (x), a list of tables of weights (w) and a table
           of biases (b).
    Output: A list of numbers corresponding to output of the ANN.
    
    >>> x = [1, 0]
    >>> w = [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> b = [[-1.1, 4.2], [-1.7, 2.5]]
    >>> y = inference(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inference(x, w, b)
    [7.35, 5.15]
    """
    pass
    
    my_list = []

    my_list.append(x)

    x = my_list

    for n in range(len(w)):
        weights = w[n]

        # make variable that calls the position of weights

        layer = create_matrix(len(x), len(weights))

        # call the function again with inputs

        for i in range(len(x)):
            for j in range(len(weights)):
                for k in range(len(x[0])):
                    layer[i][j] += x[i][k] * weights[j][k] 

        # for loops for matrix multilplication

        for i in range(len(b)):
            layer[0][i] += b[n][i]
        
        # adds each bias value

        x = layer

    return layer.pop()

def read_weights(file_name): # 1 Mark
    """
    Input: A string (file_name) that corresponds to the name of the file
           that contains the weights of the ANN.
    Output: A list of tables of numbers corresponding to the weights of
            the ANN.
    
    >>> w_example = read_weights('example_weights.txt')
    >>> w_example
    [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> w = read_weights('weights.txt')
    >>> len(w)
    3
    >>> len(w[2])
    10
    >>> len(w[2][0])
    16

    Function that reads the files, and converts it to useable input; a list of numbers that corresponds to the input of ANN.

    First, files is openedm, accessed, and closed, so it can be modified and adjusted.

    Strip and split methods are used to remove spaces and convert them into strings in lists.

    It loops through to find the hashtag, to recoginise when the new lists begins.

    Once it finds it, it gets inserted into the emmpty matrix.

    """
    pass

    f = open(file_name)
    contents = f.read()
    f.close()
 
    word = contents.strip().split()

    my_list = []

    matrix = []

    for i in range(0, len(word)):
        
        if word[i] == '#':

            matrix = []

            my_list.append(matrix)

            pass
        else:

            sentence = word[i].split(',') 
       
            row = list(map(float, sentence))

            matrix.append(row)

    return(my_list)


def read_biases(file_name): # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the biases of the ANN.
    Output: A table of numbers corresponding to the biases of the ANN.
    
    >>> b_example = read_biases('example_biases.txt')
    >>> b_example
    [[-1.1, 4.2], [-1.7, 2.5]]
    >>> b = read_biases('biases.txt')
    >>> len(b)
    3
    >>> len(b[0])
    16

    * repeats the same as the function above

    """
    pass

    f = open(file_name)
    contents = f.read()
    f.close()

    
    word = contents.strip().split()

    my_list = []

    for i in range(0, len(word)):
        
        if word[i] == '#':

            pass 
        
        else:

            sentence = word[i].split(',') 
       
            row = list(map(float, sentence))

            my_list.append(row)


    return(my_list)

def read_image(file_name): # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the image.
    Output: A list of numbers corresponding to input of the ANN.
    
    >>> x = read_image('image.txt')
    >>> len(x)
    784

    Function that converts the the numbers in the file to lists that could be manipulated.

    After opening and reading the file, numbers gets inserted into a list as a string (then converted back into integers).

    """
    pass

    f = open(file_name)
    contents = f.read()
    f.close()

    lst = contents.split()

    strings = ''.join(lst)

    row = list(map(int, strings))

    return(row)

          
def argmax(x): # 1 Mark
    """
    Input: A list of numbers (i.e., x) that can represent the scores 
           computed by the ANN.
    Output: A number representing the index of an element with the maximum
            value, the function should return the minimum index.
    
    >>> x = [1.3, -1.52, 3.9, 0.1, 3.9]
    >>> argmax(x)
    2

    A function that returns the index of an element with the highest value.

    Function loops through to find the position and thus the value of max value withint the list.

    Once it does find it, returns the value and then breaks.
    """
    pass

    for i in range(len(x)):
        if x[i] == max(x):
            return i
            break  

def predict_number(image_file_name, weights_file_name, biases_file_name): # 1 Mark
    """
    Input: A string (i.e., image_file_name) that corresponds to the image
           file name, a string (i.e., weights_file_name) that corresponds
           to the weights file name and a string (i.e., biases_file_name)
           that corresponds to the biases file name.
    Output: The number predicted in the image by the ANN.

    >>> i = predict_number('image.txt', 'weights.txt', 'biases.txt')
    >>> print('The image is number ' + str(i))
    The image is number 4

    This function brings together all the previous functions, to predict the number on the image file.

    Called each function that reads the files; biases, weights, and images, and assigned them to the variables assigned in the inference function.

    Assigned y to inference, and inputted that value to predicion to obtain the predicted number.
    """
    pass

    b = read_biases(biases_file_name)

    w = read_weights(weights_file_name)

    x = read_image(image_file_name)

    y = inference(x,w,b)

    prediction = argmax(y)

    return(prediction)




def linear(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a list of weights (w) and a bias (b).
    Output: A single number corresponding to the value of f(x) in Equation 1.

    >>> x = [1.0, 3.5]
    >>> w = [3.8, 1.5]
    >>> b = -1.7
    >>> round(linear(x, w, b),6) #linear(x, w, b)
    7.35
    """

    return sum(w[j]*x[j] for j in range(len(w))) + b


def linear_layer(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a list of 
           biases (b).
    Output: A list of numbers corresponding to the values of f(x) in
            Equation 2.
    
    >>> x = [1.0, 3.5]
    >>> w = [[3.8, 1.5], [-1.2, 1.1]]
    >>> b = [-1.7, 2.5]
    >>> y = linear_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #linear_layer(x, w, b)
    [7.35, 5.15]
    """

    return [linear(x, w[i], b[i]) for i in range(len(w))]


def inner_layer(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a 
           list of biases (b).
    Output: A list of numbers corresponding to the values of f(x) in 
            Equation 4.

    >>> x = [1, 0]
    >>> w = [[2.1, -3.1], [-0.7, 4.1]]
    >>> b = [-1.1, 4.2]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [1.0, 3.5]
    >>> x = [0, 1]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [0.0, 8.3]
    """

    return [max(linear(x, w[i], b[i]), 0.0) for i in range(len(w))]


def inference(x, w, b): # 2 Marks
    """
    Input: A list of inputs (x), a list of tables of weights (w) and a table
           of biases (b).
    Output: A list of numbers corresponding to output of the ANN.
    
    >>> x = [1, 0]
    >>> w = [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> b = [[-1.1, 4.2], [-1.7, 2.5]]
    >>> y = inference(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inference(x, w, b)
    [7.35, 5.15]
    """

    num_layers = len(w)
    
    for l in range(num_layers-1):
        x = inner_layer(x, w[l], b[l])
        
    return linear_layer(x, w[num_layers-1], b[num_layers-1])






def read_weights(file_name): # 1 Mark
    """
    Input: A string (file_name) that corresponds to the name of the file
           that contains the weights of the ANN.
    Output: A list of tables of numbers corresponding to the weights of
            the ANN.
    
    >>> w_example = read_weights('example_weights.txt')
    >>> w_example
    [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> w = read_weights('weights.txt')
    >>> len(w)
    3
    >>> len(w[2])
    10
    >>> len(w[2][0])
    16
    """

    weights_file = open(file_name,"r")
    w = []
    for line in weights_file:
        if "#" == line[0]:
            w.append([])
        else:
            w[-1].append([float(w_ij) for w_ij in line.strip().split(",")])
    
    return w


def read_biases(file_name): # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the biases of the ANN.
    Output: A table of numbers corresponding to the biases of the ANN.
    
    >>> b_example = read_biases('example_biases.txt')
    >>> b_example
    [[-1.1, 4.2], [-1.7, 2.5]]
    >>> b = read_biases('biases.txt')
    >>> len(b)
    3
    >>> len(b[0])
    16
    """

    biases_file = open(file_name,"r")
    b = []
    for line in biases_file:
        if not "#" == line[0]:
            b.append([float(b_j) for b_j in line.strip().split(",")])
    
    return b


def read_image(file_name): # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the image.
    Output: A list of numbers corresponding to input of the ANN.
    
    >>> x = read_image('image.txt')
    >>> len(x)
    784
    """

    image_file = open(file_name,"r")
    x = []
    for line in image_file:
        for x_i in line.strip():
            x.append(int(x_i))
            
    return x


def argmax(x): # 1 Mark
    """
    Input: A list of numbers (i.e., x) that can represent the scores 
           computed by the ANN.
    Output: A number representing the index of an element with the maximum
            value, the function should return the minimum index.
    
    >>> x = [1.3, -1.52, 3.9, 0.1, 3.9]
    >>> argmax(x)
    2
    """

    num_inputs = len(x)
    max_index = 0
    
    for i in range(1,num_inputs):
        if x[max_index] < x[i]:
            max_index = i
            
    return  max_index


def predict_number(image_file_name, weights_file_name, biases_file_name): # 1 Mark
    """
    Input: A string (i.e., image_file_name) that corresponds to the image
           file name, a string (i.e., weights_file_name) that corresponds
           to the weights file name and a string (i.e., biases_file_name)
           that corresponds to the biases file name.
    Output: The number predicted in the image by the ANN.

    >>> i = predict_number('image.txt', 'weights.txt', 'biases.txt')
    >>> print('The image is number ' + str(i))
    The image is number 4
    """

    x = read_image(image_file_name) 
    w = read_weights(weights_file_name)
    b = read_biases(biases_file_name)
    
    y = inference(x, w, b)


    return argmax(y)

#print(predict_number('another_image.txt', 'weights.txt', 'biases.txt'))


def read_image(file_name): # 1 Mark
    
    pass

    f = open(file_name)
    contents = f.read()
    f.close()
    lst = contents.split()
    strings = ''.join(lst)
    row = list(map(int, strings))
    return(row)


# Assigment Part 2

def flip_pixel(x):
    if x == 0:
        return 1

    if x == 1:
        return 0


    """ Flips a single pixel

    Input: An integer (x) representing a pixel in the image.
    Output: An integer representing the flipped pixel.
    
    For example:
    >>> x = 1
    >>> flip_pixel(x)
    0
    >>> x = 0
    >>> flip_pixel(x)
    1
        
    This is a list processing problem where an arithmetic operation has to be carried out
    for each element of an input list of unknown size.

    This is a function that basically flips 0 to 1 and 1 to 0; a process which is necessary once
    the pixel to be flipped is figured out. This means that if statements are necessary to flip
    0 to 1 and 1 to 0, based on the given situation. No looping is necessary.

    """

def modified_list(i,x):
    
    output = flip_pixel(x[i])
    x[i] = output
    return x

    """ Flip the pixel that is located at position i in list x 

    Input: A list of integers (x) representing the image and an integer (i) representing the position (i.e., index) of
    the pixel.
    Output: A list of integers (x) representing the modified image.
    
    >>> x = [1, 0, 1, 1, 0, 0, 0]
    >>> i = 2
    >>> modified_list(i,x)
    [1, 0, 0, 1, 0, 0, 0]


    This function modifies the specific element by flipping 0 to 1 and 1 to 0. I called the previous
    function made and literally returned the index position by assigning to an output variable.
    Again, there was no need for any loops or anything sophisicated to be implemented.
        
    """




def compute_difference(x1,x2):

    count = 0
    for i in range(len(x1)):
        if x1[i]!= x2[i]:
            count += 1
    return count

    """ Compute the total absolute difference between the adversarial image and the original image

    Input: A list of integers (x1) representing the input image and a list of integers (x2) representing the adversarial
    image.
    Output: An integer representing the total absolute difference between the elements of x1 and x2.
            
    >>> x1 = [1, 0, 1, 1, 0, 0, 0]
    >>> x2 = [1, 1, 1, 0, 0, 0, 1]
    >>> compute_difference(x1,x2)`        
    """







def second_largest(lst):

    """
    Additional function that was made to return the second highest number of a list.
    This function was used to find the second highest value within the prediction scores.
    """ 

    new_lst = sorted(lst)
    p2 = new_lst[-2]
    return lst.index(p2)



def select_pixel(x, w, b):

    """ The pixel that is to be flipped should be selected based on its overall impact on the output list of the ANN

    Input: A list of inputs (x), a list of tables of weights (w) and a table of biases (b).
    Output: An integer (i) either representing the pixel that is selected to be flipped, or with value -1 representing
    no further modifications can be made

    
    >>> x = read_image(‘image.txt’)
    >>> w = read_weights(‘weights.txt’)
    >>> b = read_biases(‘biases.txt’)
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    238
    >>> x = modified_list(pixel,x)
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    210


    In my code, I first organised the values of the original inference, which would be used to compare with the inference values (prediction scores)
    after individual pixel is flipped. The for loop was implemented to produce and obtain the impact scores of all 784 pixels. The impact scores were
    calculated by finding the absolute values between the two. All the impact scores were stored in the scores_lst. This was because if the maximum
    value of the score list is small, the predicted number would remain unchanged and thus -1 needs to be returned.

    """
    
    impact = 0
    pixel = 0
    a = argmax(inference(x,w,b)) # 4

    scores_lst = []

 
    k = second_largest(inference(x,w,b)) # 9
    prev_inf = inference(x,w,b) # [-2.166686, -5.867822999999999, -1.6730180000000001, -4.412667000000001, 5.710625000000001, -6.022383000000002, -2.1742819999999976, 0.3789300000000001, -2.267785000000001, 3.9128230000000004]



    for i in range(len(x)): # for loop to find the impact scores of 784 pixels (result after flipping each pixel)
        updating_x = x[:]
        res = modified_list(i, updating_x) # makes the x list: with pixel at i flipped        
        new_inf = inference(res,w,b) # returns
        
        if prev_inf[a] - new_inf[a] + abs(prev_inf[k]-new_inf[k]) > impact:
            impact = prev_inf[a] - new_inf[a] + abs(new_inf[k] - prev_inf[k])
            scores_lst.append(impact)            
            pixel = i
        modified_list(i, updating_x)        

    if max(scores_lst) < 0.1:

        return -1
    return pixel

x = read_image('image.txt')
w = read_weights('weights.txt')
b = read_biases('biases.txt')   





def write_image(x, file_name):

    """ Used to write the list x into a file with name file name as a 28x28 image

    Input: A list of integers (x) representing the image and a string (file name) representing the file name.
    Output: Write out each pixel represented in the list x to a file with the name file name as a 28x28 image.

    In my code, I fist opened the file. I set a few empty lists to store the iterated values which will be monitored
    (kept track) through the usage of count. I implement a for loop to keep track of how many elements have been iterated
    through and by updating it into a list once the iteration 28 times have been acheieved. I then reset the count to 0, for
    the next set of iteration. I appended the complete sublists into the lst. The second for loop was made to copy the nested
    list into a new document. The nested for loop calls each row wihtin the element and writes it into a fresh empty file as
    a string.
    
    >>> x = read_image(‘image.txt’)
    >>> x = modified_list(238,x)
    >>> x = modified_list(210,x)
    >>> write_image(x,‘new_image.txt’)      
    """

    f = open(file_name, 'w')
    
    lst = []
    temp_lst = []
    count = 0

    for i in range(len(x)):
        count += 1
        temp_lst.append(x[i])
        if count == 28:
            count = 0            
            lst.append(temp_lst)
            temp_lst = []
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            f.write(str(lst[i][j]))
        f.write('\n')

    f.close

        




def adversarial_image(image_file_name, weights_file_name, biases_file_name):


    """  Solves the adversarial image generation task.

    Input: A string (i.e., image file name) that corresponds to the image file name, a string (i.e., weights file name)
    that corresponds to the weights file name and a string (i.e., biases file name) that corresponds to the
    biases file name.
    Output: A list of integers representing the adversarial image or the list [-1] if the algorithm is unsuccesful in
    finding an adversarial image.

    
    >>> x1 = read_image(‘image.txt’)
    >>> x2 = adversarial_image(‘image.txt’,‘weights.txt’,‘biases.txt’)
    >>> if x2[0] == -1:
    ... print(‘Algorithm failed.’)
    ... else:
    ... write_image(x2,‘new_image’)
    ... q = compute_difference(x1,x2)
    ... print(‘An adversarial image is found! Total of ’ + str(q) + ‘ pixels were flipped.’)
    ...
    An adversarial image is found! Total of 2 pixels were flipped.
    Note that the behaviour of your algorithm will change

    In my code for this function, I checked using a while loop whether flipping the pixel that was requested by the
    select pixel actually impacted the final outcome. I implemented a while loop because we are not sure how many times
    the loop would need to be iterated. In round 1, pixel is selected, in round 2, the idnex position of the pixel is
    applied to produce the modified list. Then, prediction scores is obtained through inference and finally the index position
    of the maximum number is obained through the argmax function. I then checked whether the predicted number was the same
    to the one oringally obtained. If it was the same, the function would loop until a different predicted number is obtained.
    It is designed to return -1 when the impact scores decides that no impact score is sufficient enough to alter the predicted
    number. The piexls were then stored in a list so that it could be recylced when producing the modified list when writing
    the image.

    """
    
    x = read_image(image_file_name) 
    w = read_weights(weights_file_name)
    b = read_biases(biases_file_name)

    y = argmax(inference(x,w,b))

    lst = []

    c = x[:]

    check = True

    while check:

        round1 = select_pixel(c,w,b) # select the pixel with highest impact score        
        round2 = modified_list(round1,c) # x list with modified pixels
        round3 = inference(round2,w,b) # find the prediction scores with flipped pixel
        round4 = argmax(round3) # find the predicted number


        if round4 == y:
            lst.append(round1)
        elif round4 == -1:            
            print("Algorithm failed.")              
        else: # output (round4) doesn't equal y
            lst.append(round1)
            break

    for i in range(len(lst)):
        output = modified_list(lst[i],x)    
    write_image(output, 'write.txt')
    return write_image

            







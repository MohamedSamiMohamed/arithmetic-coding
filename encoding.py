import numpy as np
import cv2
# ##################################NOTE#######################################################################
# the output is some how acceptable at blocksize=10 and float64 and after than that the output won't be acceptable


# this is to read the image and put it in 1D nparray
add=0
img = cv2.imread('mama.jpeg', 0)
data = np.array(img)
print(data)
data = np.array(img).flatten()
###############
print(data)
# taking the block size and value of the float from the user
print("please enter the block size")
block_size = int(input())
print("please enter the float value like that: float16-float32-float64")
FLOAT = str(input())
#############################################################
# padd the image data depend on the value of the block size
remainder = (img.shape[0] * img.shape[1]) % block_size
if (remainder != 0):
    add = block_size - remainder
    data = np.pad(data, (0, add), mode='constant', constant_values=0)

print(data)

# this part to calculate the occurrence  of every grey degree in the image

probDict = {}
for x in range(256):
    probDict[x] = 0
for y in range(img.shape[0] * img.shape[1]):
    probDict[data[y]] = probDict[data[y]] + 1

###############################################################
print(probDict)
################################################################
# this part to evaluate the probability and remove any grey degree with probability =0
probDict.update((key, val / (img.shape[0] * img.shape[1])) for key, val in probDict.items())
probDict = {key: val for key, val in probDict.items() if val != 0}
print(probDict)
##################################################################
keys = list(probDict.keys())
##################################################################
# and this to sum probabilities
for y in range(1, len(probDict)):
    probDict[keys[y]] = probDict[keys[y]] + probDict[keys[y - 1]]
print(probDict)


# this function returns the index of a given key in a list
def find_index(list_of_keys, key):
    for x in range(0, len(list_of_keys)):
        if (list_of_keys[x] == key):
            return x
# ########################################__________ENCODING____________################################################
# this function takes sequence of data and block size and return the uppar and lower value of the tag
def arithmetic_encoding(seq, probs, blockSize, keys_list):
    upper = 1
    lower = 0
    for x in range(0, blockSize):
        j = upper
        k = lower
        upper = k + (j - k) * probs[seq[x]]
        if (seq[x] == keys_list[0]):
            lower = k + (j - k) * 0
        else:
            index = find_index(keys_list, seq[x])
            lower = k + (j - k) * probs[keys_list[index - 1]]
    return upper, lower

# ######################################___________DECODING___________________##########################################
# this function take a tag value and decode it to the right sequence
def arithmetic_decoding(code, probs, blockSize, keysList, outputSeq):
    upper = 1
    lower = 0
    for i in range(0, blockSize):
        CurrentLower = lower + (upper - lower) * 0
        CurrentUpper = lower + (upper - lower) * probs[keysList[0]]
        if (CurrentLower <= code and code < CurrentUpper):
            outputSeq[i] = keysList[0]
            upper = CurrentUpper
            lower = CurrentLower
        else:
            for j in range(1, len(keysList)):
                CurrentLower = lower + (upper - lower) * probs[keysList[j - 1]]
                CurrentUpper = lower + (upper - lower) * probs[keysList[j]]
                if (CurrentLower <= code and code < CurrentUpper):
                    outputSeq[i] = keysList[j]
                    upper = CurrentUpper
                    lower = CurrentLower
                    break

# initialize np array that will store the tag's values
arithmetic_codes = np.zeros((1, int(len(data)/ block_size)), dtype=FLOAT).flatten()

# encoding and store in the arithmetic_codes
for x in range(int(len(data)/ block_size)):
    if(x==int(len(data)/block_size)-1):
        upperTag,lowerTage= arithmetic_encoding(data[(block_size * x):(block_size * x) + (block_size)-add], probDict,
                                              block_size-add, keys)
        tagValue = (upperTag + lowerTage) / 2
        arithmetic_codes[x] = tagValue
    else:
        upperTag, lowerTage = arithmetic_encoding(data[(block_size * x):(block_size * x) + (block_size)], probDict,
                                              block_size, keys)
        tagValue = (upperTag + lowerTage) / 2
        arithmetic_codes[x] = tagValue

np.save('codes.npy', arithmetic_codes)
arithmetic_codes = np.load('codes.npy')

# initialize np array that will store the values of the decoded image
decoded_img = np.zeros((1, (img.shape[0] * img.shape[1]))).flatten()
# decode
for x in range(0, int(len(data)/ block_size)):
    if(x==int(len(data)/block_size)-1):
        arithmetic_decoding(arithmetic_codes[x], probDict, block_size-add, keys,
                            decoded_img[(block_size * x):(block_size * x) + (block_size)])
    else:
         arithmetic_decoding(arithmetic_codes[x], probDict, block_size, keys,
                        decoded_img[(block_size * x):(block_size * x) + (block_size)])


print(decoded_img, len(decoded_img))
print(data, len(data))

print(decoded_img.astype(int).reshape(img.shape[0], img.shape[1]))

cv2.imwrite('output_mama.png', decoded_img.astype(int).reshape(img.shape[0], img.shape[1]))

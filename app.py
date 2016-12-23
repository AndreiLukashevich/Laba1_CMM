import json
from hmmlearn import hmm
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('corpus.json') as f:
    corpus = json.load(f)

# init model for phoneme 'a'
model_a = hmm.GaussianHMM(n_components=3, covariance_type='diag')
# init model for phoneme 'n'
model_n = hmm.GaussianHMM(n_components=3, covariance_type='diag')

#leraning phoneme 'a', lengths leraning set l_a = len(corpus['a'])*0.8 = 179
lengths_a = []
concat_a = []
for i in range(int(len(corpus['a'])*0.8)):
    lengths_a.append(len(corpus['a'][i]))
    concat_a += corpus['a'][i]

concat_a = np.asarray(concat_a)
print('----------DATA PHONEME "a"----------')
# print (concat_a)
# print (lengths_a)
print ("lengths all set ", len(corpus['a']))
print ("lengths leraning set ", int(len(corpus['a'])*0.8))
print ("lengths test set ", (len(corpus['a']) - int(len(corpus['a'])*0.8)))
print ("list with lengths appropriate initial sequence ", len(lengths_a))
print ("long sequence (concatenate) ", len(concat_a))

# create model for phoneme 'a'
model_a.fit(concat_a, lengths_a)
#print (model_a)


#leraning phoneme 'n', lengths leraning set l_n = len(corpus['n'])*0.8 = 77
lengths_n = []
concat_n = []
for i in range(int(len(corpus['n'])*0.8)):
    lengths_n.append(len(corpus['n'][i]))
    concat_n += corpus['n'][i]

concat_n = np.asarray(concat_n)
print('----------DATA PHONEME "n"----------')
# print (concat_n)
# print (lengths_n)
print ("lengths all set ", len(corpus['n']))
print ("lengths leraning set ", int(len(corpus['n'])*0.8))
print ("lengths test set ", (len(corpus['n']) - int(len(corpus['n'])*0.8)))
print ("list with lengths appropriate initial sequence ", len(lengths_n))
print ("long sequence (concatenate) ", len(concat_n))

model_n.fit(concat_n, lengths_n)
#print (model_n)


# recognition phoneme 'a', using model_a and model_n, lengths set = 20% lengths set for phoneme 'a'
a_true = 0
a_false = 0
a_s = int(len(corpus['a'])*0.8)
a_f = int(len(corpus['a']))
a_t = int(len(corpus['a'])*0.2) + 1
for i in range(a_s, a_f):
    xa = corpus['a'][i]
    if model_a.score(xa) > model_n.score(xa):
        a_true += 1
    else:
        a_false += 1

print('----------RECOGNITION PHONEME "a"----------')
print("a_true", a_true)
print("a_false", a_false)
print("a_t",a_t)

acc_a = float(a_true) / a_t
print ("recognition accuracy for phoneme 'a', acc_a = ",acc_a)

# recognition phoneme 'n', using model_a and model_n, lengths set = 20% lengths set for phoneme 'n'
n_true = 0
n_false = 0
n_s = int(len(corpus['n'])*0.8)
n_f = int(len(corpus['n']))
n_t = int(len(corpus['n'])*0.2) + 1
for i in range(n_s, n_f):
    xn = corpus['n'][i]
    if model_n.score(xn) > model_a.score(xn):
        n_true += 1
    else:
        n_false += 1

print('----------RECOGNITION PHONEME "n"----------')
print("n_true", n_true)
print("n_false", n_false)
print("n_t",n_t)

acc_n = float(n_true) / n_t
print ("recognition accuracy for phoneme 'n', acc_n = ",acc_n)

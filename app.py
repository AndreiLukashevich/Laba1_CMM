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


# recognition phoneme 'a', using model_a, lengths set = all lengths set for phoneme 'n', len(corpus['n']) = 97 ,
# t.k. 80%*lengths set for phoneme 'a' > all lengths set for phoneme 'n'
a_true = 0
a_false = 0
for i in range(int(len(corpus['n']))):
    xa_a = corpus['a'][i]
    xn_a = corpus['n'][i]
    if model_a.score(xa_a) > model_a.score(xn_a):
        a_true += 1
    else:
        a_false += 1

print('----------RECOGNITION PHONEME "a"----------')
print("a_true", a_true)
print("a_false", a_false)

# recognition accuracy for phoneme 'a', t.k. I us 80%*lengths set for phoneme 'a' = all lengths set for phoneme 'n' =>
# lengths tes set N_a = 20%*len(corpus['n'])/80% = 24, quantity true recognition n_a = 47
N_a = 24.0
n_a = 47.0
acc_a = n_a/N_a
print ("recognition accuracy for phoneme 'a', acc_a= ",acc_a)

# recognition phoneme 'n', using model_n, lengths set = leraning set l_n = len(corpus['n'])*0.8 = 77
n_true = 0
n_false = 0
for i in range(int(len(corpus['n'])*0.8)):
    xa_n = corpus['a'][i]
    xn_n = corpus['n'][i]
    if model_n.score(xn_n) > model_n.score(xa_n):
        n_true += 1
    else:
        n_false += 1

print('----------RECOGNITION PHONEME "n"----------')
print("n_true", n_true)
print("n_false", n_false)


# recognition accuracy for phoneme 'n' lengths tes set N_n = 97-77 = 20, quantity true recognition n_n = 54
N_n = 20.0
n_n = 54.0
acc_n = n_n/N_n
print ("recognition accuracy for phoneme 'n', acc_n= ",acc_n)

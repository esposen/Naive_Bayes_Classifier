'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

PLEASE USE PYTHON 2.7

Author: Elias Posen

Date: 05.Nov.2017

'''
from __future__ import print_function
import sys
import math
import time
import random

class NaiveBayes():
    '''Naive Bayes classifier for text data.
    Assumes input text is one text sample per line.  
    First word is classification, a string.
    Remainder of line is space-delimited text.
    '''
    
    def __init__(self,train):
        '''Create classifier using train, the name of an input
        training file.
        '''
        self.vocab = {} # tracks word frequencies within classes in form {catagory:{word:frequency,...},...}
        self.unique_vocab = {} # tracks amount of unique words
        self.test_results = {} # tracks the results of the NB classification (keys:"correct" & "occurances")
        self.classes_containing = {} # track occurence of words across classes
        self.total_word_count = 0
        self.prob_type = '' # tracks of style of NB probability computation ['raw'|'mest'|'tfidf']
        self.learn(train) # loads train data, fills prob. table

    def learn(self,traindat):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''
        with open(traindat,'r') as fd:
            #Loop through the training data
            for line in fd.readlines():
                #Create dictionary such that self.vocab = {catagory:{word: frequency....}}
                words = line.split()
                id = words.pop(0)
                if not self.vocab.has_key(id): #Creates new dictionary within vocab for hiterto unseen catagory
                    self.vocab[id]={}
                    self.vocab[id]["total_words"]=0

                self.vocab[id]["total_words"] += len(words) #counts the total words for each catagory
                self.total_word_count += len(words) #counts total amount of words
                
                #Loop through all words in line and count their frequency, save according to catagory
                for word in words:
                    if self.vocab[id].has_key(word):
                        self.vocab[id][word] += 1
                    else: #First time the category has seen word
                        self.vocab[id][word] = 1

                        if not self.unique_vocab.has_key(word): #tracking of the total amount of unique words across ids
                            self.unique_vocab[word] = 1
                        if self.classes_containing.has_key(word): #tracking of the amount of classes that contain word
                            self.classes_containing[word] +=1
                        else:
                            self.classes_containing[word] = 1
    
    def printClasses(self):
        '''Prints information about classes in a table format'''
        print("############### TRAIN OUTPUT #########################")
        print("Total # Words: ", self.total_word_count)
        print("Vocab Size: ", len(self.unique_vocab) )
        print("{:^24}|{:^8}|{:^8}".format("Category","NWords","P(cat)"))
        for cat in self.vocab:
            category_total = self.vocab[cat]["total_words"]
            prob = category_total/float(self.total_word_count)
            print("{:24}|{:^8}|{:8.3f}".format(cat,category_total,prob))

    def runTest(self, testdat, probType):
        '''Loads data for testing and test according to prabability type'''
        
        self.test_results={category: {"occurances":0,"correct":0} for category in self.vocab} #setup to store test results
        self.prob_type = probType

        with open(testdat,'r') as td:
            for line in td.readlines():
                category_probs = [] #Stores probability of "line" being in each catagory
                words = line.split()
                real_category = words.pop(0) #Saves actual catagory for comparison
                self.test_results[real_category]["occurances"] += 1

                # Compute proability of "line" being in each catagory
                for category in self.vocab:
                    category_total = self.vocab[category]["total_words"]
                    running_prob = category_total/float(self.total_word_count)
                    for word in words:
                        # Check what method of probability computation
                        if self.prob_type == 'raw':
                            running_prob *= self.rawProb(word, category, category_total)
                        elif self.prob_type == 'mest':
                            running_prob += math.log(self.mestProb(word, category, category_total))#Using sum of logs to avoid float underflow
                        elif self.prob_type == 'tfidf':
                            running_prob += math.log(self.tfidfProb(word, category, category_total))#Using sum of logs to avoid float underflow
                    category_probs.append(running_prob) #Save probability for each catagory
            
                guess_category = self.vocab.keys()[argmax(category_probs)]
                #Check guess based on NB classifier against real catagory
                if guess_category == real_category:
                    self.test_results[guess_category]["correct"] += 1

    def printTestResults(self):
        print("\n############### TEST OUTPUT #########################")
        print("Probability Type: %s"  % self.prob_type)
        print("{:^24}|{:^8}|{:^5}|{:^8}".format("Category","NCorrect","N","%Correct"))
        avg = 0 
        for category in self.test_results:
            correct = self.test_results[category]["correct"]
            occur = self.test_results[category]["occurances"]
            print("{:24}|{:^8}|{:^5}|{:^8.3f}".format(category, correct, occur,  100*(correct/float(occur))))
            avg += 100*(correct/float(occur))
        print("Average Accuracy:", avg/20)

    def rawProb(self, word, category, category_total):
        ''' Computes and returns raw probability of word given catagory'''
        if self.vocab[category].has_key(word):
            return self.vocab[category][word]/float(category_total)
        return 0

    def mestProb(self, word, category, category_total):
        '''Computes and returns m-estimate probability of word given catagory'''
        if self.vocab[category].has_key(word):
            return (self.vocab[category][word] + 1.0) / (float(category_total + len(self.unique_vocab)))
        return 1.0 / (float(category_total + len(self.unique_vocab)))

    def tfidfProb(self, word, category, category_total):
        '''Computes and returns tf-idf probability of word given catagory'''
        if self.unique_vocab.has_key(word):
            idf = math.log((len(self.vocab) + 2.0 )/ (1.0+self.classes_containing[word])) # 2 added to numerator to avoid idf being 0
        else:
            idf = math.log(len(self.vocab))
        if self.vocab[category].has_key(word):
            tf = (self.vocab[category][word] + 0.1)/(float(category_total))
        else:
            tf = 0.1/(float(category_total)+len(self.unique_vocab))
        return abs(idf * tf)

def argmax(lst):
    '''Returns index of element with highest value'''
    return lst.index(max(lst))
    
def main():

    if len(sys.argv) != 4:
        print("Usage: %s trainfile testfile ['raw'|'mest'|'tfidf']" % sys.argv[0])
        sys.exit(-1)
    t= time.time()#tracks run time of program

    #Learn, print classes, run test, print test results. 
    nbclassifier = NaiveBayes(sys.argv[1])
    nbclassifier.printClasses()
    if sys.argv[3] in ['raw','mest', 'tfidf']: #checks proability type is valid
        nbclassifier.runTest(sys.argv[2], sys.argv[3])
    else:
        print("Probability version ", sys.argv[3], " not defined....TERMINATING PROGRAM")
        sys.exit(-1)
    nbclassifier.printTestResults()
    print("\nRUN TIME:", time.time()-t)

if __name__ == "__main__":
    main()

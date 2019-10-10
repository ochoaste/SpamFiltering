import numpy as np
import sklearn
import os
import re
import string
import io
import random 

# location of index; program later needs to leave to access data directory
path = '/home/ochoaste/Desktop/395/trec07p/full/'


f1 = open(path+'index')
lines = f1.readlines()
f1.close() # close immediately
#print(lines)
labels = dict()
for l in lines:
    (tag, file_id) = l.split()
    if tag in labels.keys():
        labels[tag].append(file_id)
    else:
        labels[tag] = [file_id]
print(labels.keys())
print('Total number of spam emails: ', len(labels['spam']))
print('Total number of ham emails: ', len(labels['ham']))


def convert_byte_to_words(line):
    
    text = str(line, 'utf-8', 'ignore')
    text = re.sub(r'[^\w\s]','',text) # removes punctuation, whitespace
    words = text.lower().split()
    # can remove stop words here
    return words
    
    
def clean_data(all_lines):
    # some cleaning may still need to be done
    
    only_content = []
    words_dict = dict()
    for i in range(len(all_lines)):
            # converts byte code to a string
            current_line = str(all_lines[i], 'utf-8', 'ignore')
            # 'ignore' prevents it from throwing a utf error
            if (current_line.startswith('Lines:')):
                s = current_line.split()
                n = int(s[1])
                only_content = all_lines[:n:-1] # reads LAST n lines in file
                for l in only_content:
                    words = convert_byte_to_words(l)
                    for w in words:
                        if w not in words_dict.keys():
                            words_dict[w] = 1
                        else:
                            words_dict[w] += 1
                break
    return words_dict


# reading all training files for spam or ham #
def reading_files(training_fls):

    all_contents = dict()
    words = 0
    for t in training_fls:
        fname = os.path.join(path, t) # changes paths to get data
        x_file = open(os.path.join(path, t), 'rb') # opens in byte code to read it   
        lines = x_file.readlines()
        x_file.close()
        local_words = clean_data(lines)
        for lw in local_words.keys():
            if lw not in all_contents.keys():
                all_contents[lw] = local_words[lw]
            else:
                all_contents[lw] += local_words[lw]
            words = local_words[lw]    
    return all_contents
    
def email_Type(file):
    path = '/home/ochoaste/Desktop/395/trec07p/full/'
    f1 = open(path+'index')

    templine = f1.readline().split()
    shpam = templine[0]
    templine = templine[1]
    
    while templine != file:
        templine = f1.readline().split()
        shpam = templine[0]
        templine = templine[1]
        
    email_type = shpam
    f1.close()


    if email_type == 'ham':
        return 1
    else:
        return 0
'''        
    tempfile = f1.readlines()
    f1.close()

    for current in tempfile:
        if current == file:
            email_type = tempfile[current-1]
            if email_type == 0: #spam
                return 0
            else:
                return 1 #ham
    '''

    
training_files_s = labels['spam'][:1000] # first 1000 spam files
training_files_h = labels['ham'][:1000] # first 1000 ham files
all_contents_spam = reading_files(training_files_s)
all_contents_ham = reading_files(training_files_h)


'''
# reading all training spam files #
all_contents_spam = dict()
spam_words = 0
for t in training_files_s:
    fname = os.path.join(path, t) # changes paths to get data
    x_file = open(os.path.join(path, t), 'rb') # opens in byte code to read it   
    lines = x_file.readlines()
    x_file.close()
    local_words = clean_data(lines)
    for lw in local_words.keys():
        if lw not in all_contents_spam.keys():
            all_contents_spam[lw] = local_words[lw]
        else:
            all_contents_spam[lw] += local_words[lw]
        spam_words = local_words[lw]
    #print(local_words)
print(all_contents_spam)


# reading all training ham files #
all_contents_ham = dict()
ham_words = 0
for t in training_files_h:
    fname = os.path.join(path, t)
    x_file = open(os.path.join(path, t), 'rb')    
    lines = x_file.readlines()
    x_file.close()
    local_words = clean_data(lines)
    for lw in local_words.keys():
        if lw not in all_contents_ham.keys():
            all_contents_ham[lw] = local_words[lw]
        else:
            all_contents_ham[lw] += local_words[lw]
        ham_words += local_words[lw]
    #print(local_words)
print(all_contents_ham)
'''


def main():
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    #### building vocabulary ###
    all_words = dict()
    total_words = 0
    # combine both spam and ham dictionary into master vocab
    for w in all_contents_spam.keys():
        all_words[w] = all_contents_spam[w]
        total_words += all_contents_spam[w]
    for w in all_contents_ham.keys():
        if w in all_words.keys():
            all_words[w] += all_contents_ham[w]
        else:
            all_words[w] = all_contents_ham[w]
        total_words += all_contents_ham[w]
    print('All unique words: ', len(all_words))
    print('total words: ', total_words)


    list_of_test_files = labels['spam'][1001:1021] + labels['ham'][1001:1021]
    #list_of_test_files = random.shuffle(list_of_test_files)
    #print(list_of_test_files)
    for file in range(len(list_of_test_files)):
        #test_file = labels['spam'][35810] # testing with a spam
        print(file)
        test_file = list_of_test_files[file]
        print(test_file)
        fname = os.path.join(path, test_file)
        x_file = open(os.path.join(path, test_file), 'rb')    
        lines = x_file.readlines()
        x_file.close()
        local_words = clean_data(lines)
        # for now, probability of spam = probability of ham = 0.5 as there are equal number of cases in the training file


        # determining the probability of being spam
        probability_spam = 0.5
        for lw in local_words.keys():
                if lw in all_contents_spam.keys():
                    probability_spam *= (all_contents_spam[lw]/all_words[lw])
                    # ^^ calculates the probability function, i.e. P(A|B)
        probability_ham = 0.5
        for lw in local_words.keys():
                if lw in all_contents_ham.keys():
                    probability_ham *= (all_contents_ham[lw]/all_words[lw])


        print('spam probability: ', probability_spam)
        print('ham probability: ', probability_ham)
        
        
        if(probability_spam > probability_ham):
            print('We THINK this is spam!')
            if email_Type(test_file) == 0:
               print('This is actually spam', '\n')
               true_negative += 1
            else:
               print('This is actually ham', '\n')
               false_negative += 1
            
        else:
            print('We THINK this is ham :)')
            if email_Type(test_file) == 1:
               print('This is actually ham', '\n')
               true_positive += 1
                          
            else:
               print('This is actually spam', '\n')
               false_positive += 1
            
    print('Total true negatives: ', true_negative)
    print('Total true positives: ', true_positive)
    print('Total false negatives: ', false_negative)
    print('Total false positives: ', false_positive)



if __name__== '__main__':
    main()
import os
import re
import string

import nltk
from PyPDF2 import PdfReader
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


from nltk.stem import PorterStemmer, WordNetLemmatizer
#stemmer = PorterStemmer()
lem = WordNetLemmatizer()



# paths = 'resum'
# pdff = [os.path.join(paths,f) for f in os.listdir(paths) if f.endswith('.pdf')]
# pdff = pdff[:10]
stp = stopwords.words('english')
addw = ['job', 'description', 'qualification', 'responsibility', 'requirement', 'skills']
stp.extend(addw)

def extract(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text()
    return preprocess(text)


def preprocess(text):
    text = text.lower()
    tk = custom(text)
    #text = ' '.join(i for i in tk)
    #print(text)
    filtered = [w for w in tk if w not in stp and w!='.']
    #stemd = [' '.join(stemmer.stem(w) for w in filtered)]
    stemd = [' '.join(lem.lemmatize(w) for w in filtered)]
    return stemd

#jobd = "Looking for a person having experience with C C++ Python Java NLP Machine Learning Brain Neural Network MERN Linux"


def custom(text):
    pattern = r'\w*[a-zA-Z\d.+*-/#]+\w*'
    tokens = re.findall(pattern, text)
    # print(tokens)
    return tokens

def keyextract(text):
    tk = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tk)
    kw = ' '.join(word for word, pos in tag if pos.startswith('NN'))
    return kw


#tf = TfidfVectorizer()
cv = CountVectorizer(tokenizer=custom,stop_words='english')


def func(jobd,pdff):
        jobd = keyextract(jobd)
        jobd = preprocess(jobd)
        #print(jobd)
        # jobv = tf.fit_transform([jobd])
        jobv = cv.fit_transform(jobd)
        # print(jobv.toarray())

        rank = []

        for pdf in pdff:
            restext = extract(pdf)
            #print(restext)
            fname = os.path.basename(pdf)
            #resvec = tf.transform(restext)
            resvec = cv.transform(restext)
            x=cosine_similarity(jobv,resvec)[0][0]
            rank.append((fname,x))

        rank.sort(key=lambda x: x[1], reverse=True)
        return rank
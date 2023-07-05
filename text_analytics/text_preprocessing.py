# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:54:10 2023

@author: Sujith Thillayampalam
"""

import requests
import re
from bs4 import BeautifulSoup as bs
import string
from nltk.corpus import treebank
from nltk.classify import NaiveBayesClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.stem import WordNetLemmatizer

url='https://www.bbc.com/news/business-62390097'
url='https://www.investing.com/magazine/these-are-the-worlds-youngest-billionaires-3/'
url = 'https://www.songmeaningsandfacts.com/katy-perrys-dark-horse-lyrics-meaning/'

def extract_web_text(url, tags=['p'], filterAttr={}):
    text = ''
    html = None
    soup = None
    title = None
    body = None
    
    try:
        html = requests.get(url, timeout=60).content
        soup = bs(html, 'html.parser')
        title = soup.find('head')
        body = soup.find('body')
    
    except:
        print("Connectivity error with news url: %s" % url)
        return ''
    
    if title != None:
        for c in title.find_all(tags, filterAttr):
            text += c.text.strip()
    
    if body != None:
        for c in body.find_all(tags, filterAttr):
            content = re.sub(r'[\t\r\n]', r' ', c.text.strip())
            #print("---" + p.text + "---" + p.name)
            
            if len(content) < 19 or (len(content) > 18 and re.search(r"\s", content)):
                text += ' ' + content
        
    return text
    

contraction_map = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


def expand_contractions(sentence, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


def clean_text_content(content):
    
    content = re.sub(r"[\'|’]", "'", content.strip())
    content = re.sub(r"[“|”]", "'", content.strip())
    content = re.sub(r"[?|$|&|*|%|@|(|)|~|©|\\|™]", r' ', content)
    content = re.sub(r'[\s]+', r' ', content)
    content = expand_contractions(content, contraction_map)
    
    return content


import nltk
from nltk.corpus import wordnet as wn

def lemmatize_text(content, tagger, lemmatizer):
    sentences = nltk.sent_tokenize(content)
    lemmatized_text = ''
    
    for s in sentences:
        tokens = [t.strip() for t in nltk.word_tokenize(s)]
        
        tagged_token = tagger.tag(tokens)
        lemmatized_tokens = []
        
        for word, tag in tagged_token:
            custom_tag = None
            
            if tag.startswith('J'):
                custom_tag = wn.ADJ
            elif tag.startswith('V'):
                custom_tag = wn.VERB
            elif tag.startswith('N'):
                custom_tag = wn.NOUN
            elif tag.startswith('R'):
                custom_tag = wn.ADV
            #else:
                #custom_tag = None
            
            if custom_tag:
                lemmatized_tokens.append(lemmatizer.lemmatize(word.lower(), custom_tag))
            else:
                lemmatized_tokens.append(word.lower())
                
        lemmatized_text += " ".join(lemmatized_tokens)
            
    return lemmatized_text


def remove_special_characters(text):
    tokens = nltk.word_tokenize(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_functional_words(text, stopword_list):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

### testing

url = 'https://edition.cnn.com/2023/06/14/world/enceladus-ocean-phosphorus-scn/index.html'

tags = ['h1', 'h2', 'h3', 'h4', 'p', 'time', 'span']
paragraphs = extract_web_text(url, tags)

cleaned_text = clean_text_content(paragraphs)

wnl = WordNetLemmatizer()
tagged_data = treebank.tagged_sents()
nb_tagger = ClassifierBasedPOSTagger(train=tagged_data, classifier_builder=NaiveBayesClassifier.train)
stopwords = nltk.corpus.stopwords.words('english')

lemmatized_text = lemmatize_text(cleaned_text, nb_tagger, wnl)
lemmatized_text = remove_special_characters(lemmatized_text)
lemmatized_text = remove_functional_words(lemmatized_text, stopwords)

from datetime import datetime, date
import snscrape.modules.twitter as sntwitter
import pandas as pd

to_date = datetime(2023, 6, 30)
from_date = datetime(to_date.year - 1, to_date.month, to_date.day)

def find_tweets_with_news_links(handler_list, from_date, to_date):
    #handler_name = 'theisland_lk'
    tquery = "%s lang:en until:%s since:%s" % (('@' + handler_list), to_date.strftime("%Y-%m-%d"), from_date.strftime("%Y-%m-%d"))
    tweets = sntwitter.TwitterSearchScraper(tquery).get_items()
    
    #count = 0
    filtered_tweets = []
    web_url_regex = "(https?://t\.co/([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)"
    
    for t in tweets:
        urls = re.findall(web_url_regex, t.rawContent)
        news_url = ''
        for x in urls:
            news_url = x[0].strip()
            filtered_tweets.append([t.id, handler_list, t.date, news_url])
                
        #count += 1
        #if count == 10:
        #    break;

    headers = ['id', 'source', 'date', 'url']
    
    return pd.DataFrame(filtered_tweets, columns=headers)

def extract_clean_text(handler_df, links_df, start=0, stop=1000):
    #links_df["content"] = ""
    handler_map = {}
    #tags = ['h1', 'h2', 'h3', 'h4', 'time', 'div']
    
    for inx, row in handler_df.iterrows():
        handler_attr = {}
        handler_attr["name"] = row["name"]
        handler_attr["tags"] = row["content_tags"].split()
        handler_map[row["handlers"]] = handler_attr
    
    for inx, row in links_df.iterrows():
        if inx < start:
            continue;
            
        if inx > stop:
            break;
        
        handler_attr = handler_map[row["source"]]
        content = extract_web_text(row["url"], handler_attr["tags"])
        content = clean_text_content(content)
        print(inx)
        
        if handler_attr["name"] in content:
            links_df.at[inx, "content"] = content
    
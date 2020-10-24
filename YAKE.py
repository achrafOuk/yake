from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re
from WordStat import stats
from collections import defaultdict
import math
from termScore import TermScore
from statistics import median,mean,stdev
from jellyfish import jaro_similarity
from helper import helper
"""YAKE keyphrase extraction model.
Statistical approach to keyphrase extraction described in:
* Ricardo Campos, Vítor Mangaravite, Arian Pasquali, Alípio Mário Jorge,
  Célia Nunes and Adam Jatowt.
  YAKE! Keyword extraction from single documents using multiple local features.
  *Information Sciences*, pages 257-289, 2020.
"""
class yake(helper):
    """
    @:param:
    text
    keywordsNumber:number of keywords
    DuplicatePrama:the phram where duplicate keywords is acceptable
    """
    def __init__(self, text,DuplicatePrama=.6,window=2):
        self.text = text.replace('\n', ' ')
        self.stopWord = stopwords.words('english')
        self.window=window
        self.Chuncks = None
        self.terms = defaultdict(stats)
        self.tokens = defaultdict(stats)
        self.termsCalcule = defaultdict(TermScore)
        self.WordScore = dict()
        self.candidateKeywords = defaultdict(stats)
        self.cooccur = {}
        # check if word is either an Anacronm or Uppercase
    #create listes of chuncked
    def chunks(self):
        chunks=[]
        chunk=[]
        sentences=sent_tokenize(self.text)
        for sentence in sentences:
            for word in word_tokenize(sentence):
                if word in "./()[]?!:,;":
                    if len(chunk)!=0:
                        chunks.append(chunk)
                        chunk=[]
                elif word not in "’'":
                    chunk.append( word )
        if len(chunk) != 0:
            chunks.append(chunk)
        return chunks

    def preprocessing(self):
        self.Chuncks = self.chunks()
        self.chunckDict = []
        wordDict=dict()
        for chunck in self.Chuncks:
            wordDict=dict()
            for word in range(len(chunck)):
                #get the the set of words
                if chunck[word].lower() not in self.tokens:
                    self.tokens[chunck[word]]
                wordDict[chunck[word]] = self.getNameTag(chunck[word], word)
            self.chunckDict.append(wordDict)
        #self.Chuncks = [[tokens.lower() for tokens in chunk] for chunk in self.Chuncks]
        return self.chunckDict

    def compute_term_statistics(self):
        sentencesL = sent_tokenize(self.text)
        for chunk in self.Chuncks:
            for index in range(len(chunk)):
                if chunk[index].lower() not in self.stopWord and len(chunk[index]) >= 3:
                    self.terms[chunk[index]].TF += 1
                    for j in range(self.window):
                        try:
                            if (self.Chuncks[index], self.Chuncks[j - index]) not in self.cooccur.keys():
                                self.cooccur[(chunk[index], chunk[j - index])] = 0
                            if self.occurance(chunk,chunk[index],chunk[j - index] ):
                                self.cooccur[(chunk[index], chunk[j - index])] += 1
                        except:
                            pass
                        try:
                            if (chunk[index], chunk[j + index]) not in self.cooccur.keys():
                                self.cooccur[(chunk[index], chunk[j - index])] = 0
                            if self.occurance(chunk, chunk[index], chunk[j + index]):
                                self.cooccur[(chunk[index], chunk[j - index])] += 1
                        except:
                            pass
                    self.terms[chunk[index]].TF_a = self.get_word_tags(self.chunckDict,chunk[index],"a")
                    self.terms[chunk[index]].TF_a = self.get_word_tags(self.chunckDict, chunk[index], "U")
                    self.terms[chunk[index]].offsets_sentences = self.getSumIndexSents(sentencesL, chunk[index])

        return self.terms

    def calcule_DL(self, dicte, word):
        DL = 0
        EnumElemes = 0
        for key in dicte.keys():
            if key[0] == word:
                EnumElemes += 1
                DL += 1
        return DL, EnumElemes

    def calcule_DR(self, dicte, word):
        DR = 0
        EnumElemes = 0
        for key in dicte.keys():
            if key[0] == word:
                EnumElemes += 1
                DR += 1
        return DR, EnumElemes

    def Features_computation(self):
        sentencesL = sent_tokenize(self.text)
        validTFs = [self.terms[term].TF for term in self.terms if term.lower() not in self.stopWord]
        maxTF = max([self.terms[term].TF for term in self.tokens])
        avgTF = mean(validTFs)
        stdTF = stdev(validTFs)
        for word in self.terms:
            if word.lower() not in self.stopWord and len(word)>=3:
                #print(":-:-:", word)
                Tfa = self.terms[word].TF_a
                TfU = self.terms[word].TF_U
                TF = self.terms[word].TF
                self.termsCalcule[word].TCase = max(Tfa, TfU) / 1 + math.log(TF)
                #print(word,self.termsCalcule[word].TCase)
                self.termsCalcule[word].TPos = math.log(3 + median( self.getPosIndexSents(sentencesL, word ) ))
                self.termsCalcule[word].TFNorm = self.terms[word].TF / (avgTF + stdTF)
                self.termsCalcule[word].TSent = len( self.getPosIndexSents(sentencesL, word )) / len(sent_tokenize(self.text))
                try:
                    DL = self.calcule_DL(self.cooccur,self.cooccur(word))[1]/self.calcule_DL(self.cooccur,self.cooccur(word))[0]
                except:
                    DL =0
                try:
                    DR = self.calcule_DR(self.cooccur, self.cooccur(word))[1] / self.calcule_DR(self.cooccur, self.cooccur(word))[0]
                except:
                    DR =0
                self.termsCalcule[word].TRel = 1+(DL+DR) * ( self.terms[word].TF / maxTF )

    def term_score(self):
        for word in self.tokens:
            for word in self.tokens:
                if word.lower() not in self.stopWord and len(word) >= 3:
                    # (TPos ∗ TRel) / (TCase + ((TFNorm + TSent) / TRel))
                    TPos = self.termsCalcule[word].TPos
                    TRel = self.termsCalcule[word].TRel
                    TCase = self.termsCalcule[word].TCase
                    TFNorm = self.termsCalcule[word].TFNorm
                    TSent = self.termsCalcule[word].TSent
                    try:
                        self.WordScore[word] = (TPos * TRel) / (TCase + ((TFNorm + TSent) / TRel))
                    except:
                        self.WordScore[word] = 0
    def __existe(self, word):
        for items in self.chunckDict:
            if word in items and items[word] in ['p', 'U', 'a']:
                return True
        return False
    def start_or_end_with_stop_word(self, sentence):
        listeOFword = word_tokenize(sentence)
        if (listeOFword[0].lower() in self.stopWord) or (listeOFword[-1].lower() in self.stopWord):
            return True
        return False
    def ngrams_generation(self, n=3):
        sentences = sent_tokenize(self.text)
        chunks =  self.Chuncks
        #chunks = [[re.sub(r'[\W]+', ' ', tokens) for tokens in chunk] for chunk in chunks]

        #chunks = [[tokens.lower() for tokens in chunk] for chunk in chunks]
        for tokens in chunks:
            for i in range(len(tokens)):
                #print(":-:",tokens[i] , self.__existe(tokens[i]))
                cand = ""

                if self.__existe(tokens[i]):
                    for j in range(n):
                        try:
                            if "." not in " ".join(tokens[i:i + j]+1) or "," not in " ".join(tokens[i:i + j+1]):
                                cand = " ".join(tokens[i:i + j]) + " "
                                if (not self.start_or_end_with_stop_word(cand)):
                                    self.candidateKeywords[cand].KF += 1
                                    #print(cand)
                        except:
                            pass

    def ngrams_generation(self, n=3):
        chunks = self.Chuncks
        #chunks = [[re.sub(r'[\W]+', ' ', tokens) for tokens in chunk] for chunk in chunks]
        for tokens in chunks:
            for i in range(len(tokens)):
                if self.__existe(tokens[i]):
                    cand = ""
                    for j in range(n):
                        try:
                            cand += tokens[i+j]+" "
                            if (not self.start_or_end_with_stop_word(cand)):
                                self.candidateKeywords[cand].KF += 1
                        except:
                            pass

    def Proba(self, term1, term2):
        cooccure = 0
        occurance = 0
        for chunck in self.Chuncks:
            if term1 in chunck:
                occurance += 1
            if term1 in chunck and term2 in chunck and abs(chunck.index(term2) - chunck.index(term1)) == 1:
                cooccure += 1
        return cooccure / occurance

    def candidate_keyword_score(self):
        for candidats in self.candidateKeywords:
            tokens = candidats.split(" ")
            prod_S = 1
            sum_S = 0
            #tokens = tokens[:-2]
            tokens = tokens[:-1]
            for i in range(len(tokens)):
                #print( self.WordScore["Keyphrase"])
                if tokens[i] in self.WordScore and tokens[i].lower() not in self.stopWord:
                    prod_S *= self.WordScore[tokens[i]]
                    sum_S += self.WordScore[tokens[i]]
                else:
                    try:
                        probBefore = self.Proba(tokens[i], tokens[i - 1])
                        probAfter = self.Proba(tokens[i], tokens[i + 1])
                        BigramProbability = probBefore * probAfter
                    except:
                        BigramProbability = 0
                    prod_S *= 1 + (1 - BigramProbability)
                    sum_S -= (1 - BigramProbability)
                self.candidateKeywords[candidats].Score = prod_S / (self.candidateKeywords[candidats].KF * (sum_S + 1))

        self.candidateKeywords = sorted(self.candidateKeywords.items(), key=lambda k: k[1].Score)
        for i in self.candidateKeywords:
            print(i[0],i[1].Score)
    def word_deduplication(self,threshold=.8):
        keywords = []

        for index,item in enumerate(self.candidateKeywords):
            keywords.append(item[0])
            if index>0:
                break
        for candidate in self.candidateKeywords:
            skip = False
            for key in keywords:
                if jaro_similarity(key.lower(),candidate[0].lower()) > threshold:
                    skip = True
                    break
            if not skip:
                keywords.append(candidate[0])
        print(keywords)
        #print( sorted([x[0] for x in self.candidateKeywords if x[0] in keywords], key=lambda k:k))
        return sorted([x[0] for x in self.candidateKeywords if x[0] in keywords], key=lambda k:k)

    #sortedlist = sorted([x for x in liste if x%2!=1], key=lambda k:k)
    def get_keyword(self,n=5):
        self.preprocessing()
        self.compute_term_statistics()
        self.Features_computation()
        self.ngrams_generation()
        self.term_score()
        self.candidate_keyword_score()
        keywords=self.word_deduplication()
        if len(keywords)>n:
            return keywords[:]
        return keywords[:n]



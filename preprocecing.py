import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from segtok.segmenter import split_single, split_multi
from nltk.corpus import stopwords
import re
from WordStat import stats
from collections import defaultdict
import math
from termScore import TermScore
from statistics import median,mean,stdev
from jellyfish import jaro_similarity

"""YAKE keyphrase extraction model.
Statistical approach to keyphrase extraction described in:
* Ricardo Campos, Vítor Mangaravite, Arian Pasquali, Alípio Mário Jorge,
  Célia Nunes and Adam Jatowt.
  YAKE! Keyword extraction from single documents using multiple local features.
  *Information Sciences*, pages 257-289, 2020.
"""
class preprocessing:
    def __init__(self,text):
        #lower the case of all the letters in the text
        self.text=text = text.replace('\n',' ')
        self.stoplist = None
        self.language = None
        self.stopWord = stopwords.words('english')
        self.DuplicatePrama=0.8
        self.chunckDict=dict()
        self.Chuncks=None
        self.temrs = defaultdict(stats)
        self.window=2
        self.cooccur={}
        ########A verfier
        self.tokens=defaultdict(stats)
        ########
        self.terms = defaultdict(stats)
        self.termsCalcule= defaultdict(TermScore)
        self.WordScore=dict()
        self.candidateKeywords = defaultdict(stats)
    #check if word is either an Anacronm or Uppercase
    def isAnacronimOrUpperCase(self,word):
        wordOfUppercases = 0
        for letter in word:
            if letter.isupper():
                wordOfUppercases+=1
        if wordOfUppercases==len(word):
            return 0
        if word[0].isupper() and wordOfUppercases==1:
            return 1
        else:
            return 2
    # check is the word a mail
    def isMail(self,word):
        if re.match("([\w][_]?[.]?)*[@]([\w][.]?)+",word):
            return True
        return False
    #create a liste of chuncked
    def chunks(self,sentences):
        chunks=[]
        chunk=[]
        for sentence in sentences:
            for word in word_tokenize(sentence):
                if word in "./()[]?!:":
                    if len(chunk)!=0:
                        chunks.append(chunk)
                        chunk=[]
                elif word not in " ’'":
                    chunk.append( word )
        return chunks
   #get the tage name of the word
    def getNameTag(self,word,pos):
        #return d if word is a number
        if re.match("[\d]+(.[\d]+)?|[\d]+,[\d]+",word):
            return "d"
        #A term only formed by uppercase chars
        elif self.isAnacronimOrUpperCase(word)==0:
            return "a"
        #return u if the word is either a mail or contains a number or symbols
        elif self.isMail(word) or re.search("(\w+)(\d+)(\w+)?|(\d+)(\w+)",word) or re.search("[#|:]+",word):
            return "u"
        # return U if the word is an acronim or an upper case word that does not in the beggining of the sentence
        elif self.isAnacronimOrUpperCase(word)==1 and pos!=0:
            return "U"
        #return else otherwise
        else:
            return "p"
    def preprocessing(self):
        #dévisé le texte entre phrases
        sentences = split_multi(self.text)
        #obtenir les morseaux de texte
        self.Chuncks = self.chunks(sentences)
        self.chunckDict = []
        wordDict=dict()
        for chunck in self.Chuncks:
            wordDict=dict()
            for word in range(len(chunck)):
                #get the the set of words
                if chunck[word].lower() not in self.tokens:
                    self.tokens[chunck[word].lower()]
                wordDict[chunck[word]] = self.getNameTag(chunck[word], word)
                self.chunckDict.append(wordDict)
    """
    @output  List of terms and corresponding statistics, cooccur matrix
    """
    def get_word_tags(self,word,tag):
        compte=0
        for i in range(len(self.chunckDict)):
            try:
                if(self.chunckDict[i][word]== tag):
                    compte+=1
            except:
                pass
        return compte
    def occurance(self,sentence,word1,word2):
        occur=False
        for i in range(len(sentence)):
            if(sentence.index(word1)-sentence.index(word2)<=self.window):
                return True
        return occur
    def check_key_exists(self,dicte,key):
        return True if key in dicte.keys() else False

    def __getSumIndexSents(self,sentences,word):
        Sumpos=0
        for sentence in sentences:
            if word in sentence:
                Sumpos += sentences.index(sentence)
        return Sumpos
    def compute_term_statistics(self):
        sentencesL = sent_tokenize(self.text)
        sentencesL = list(map(lambda sente:sente.lower(),sentencesL))
        sentences = split_multi(self.text)
        chuncks = self.chunks(sentences)
        for chunck in chuncks:
            for word in range(len(chunck)):
                if chunck[word] not in self.stopWord and len(chunck[word]) > 3:
                    # calcule the TF of word
                    self.terms[chunck[word]].TF += 1
                    # calcule the sum of position of sentence where word existe
                    self.terms[chunck[word]].offsets_sentences = self.__getSumIndexSents(sentencesL, chunck[word])
                    for j in range(self.window):
                        try:
                            if (chunck[word], chunck[j-word]) not in self.cooccur.keys():
                                self.cooccur[(chunck[word], chunck[j - word])] = 0
                            elif self.occurance(chunck, chunck[word], chunck[j-word]):
                                self.cooccur[(chunck[word], chunck[j - word])] += 1
                        except:
                            pass
                        try:
                            if (chunck[word], chunck[j + word]) not in self.cooccur.keys():
                                self.cooccur[(chunck[word], chunck[j + word])] = 0
                            elif self.occurance(chunck, chunck[word], chunck[j+word]):
                                self.cooccur[(chunck[word], chunck[j + word])] += 1
                        except:
                            pass

    # calcule the occurance a word appers in right

    def calcule_DR(self,dicte,word):
        DR=0
        EnumElemes = 0
        for key in dicte.keys():
            if key[0] == word:
                EnumElemes += 1
                DR+=1
        return DR,EnumElemes

    # calcule the occurance a word appers in left
    def calcule_DL(self,dicte,word):
        DL=0
        EnumElemes = 0
        for key in dicte.keys():
            if key[0] == word:
                EnumElemes += 1
                DL+=1
        return DL,EnumElemes
    def Features_computation(self):

        for word in self.tokens:
            Tfa= self.get_word_tags(word,"a")
            TfU = self.get_word_tags(word, "U")
            #calcule TCase
            self.termsCalcule[word].TCase = max(Tfa,TfU) / math.log( 1+ math.log(self.terms[word].TF) )
            #TPos
            self.termsCalcule[word].TPos = math.log( 3 + median(self.terms[word].offsets_sentences))
            #TFNorm
            validTFs = [ self.terms[term].TF for term in self.tokens if not self.stopWord]
            avgTF = mean(validTFs)
            stdTF = stdev(validTFs)
            self.termsCalcule[word].TFNorm = self.terms[word].TF / (avgTF + stdTF)
            #len( split_multi(self.text))
            self.termsCalcule[word].TSent = self.terms[word].offsets_sentences/len( split_multi(self.text))

            #TRel
            maxTF = max([ self.terms[term].TF for term in self.tokens])
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
            #(TPos ∗ TRel) / (TCase + ((TFNorm + TSent) / TRel))
            TPos= self.termsCalcule[word].TPos
            TRel = self.termsCalcule[word].TRel
            TCase = self.termsCalcule[word].TCase
            TFNorm = self.termsCalcule[word].TFNorm
            TSent = self.termsCalcule[word].TSent
            try:
                self.WordScore[word] = (TPos * TRel) / (TCase + ((TFNorm + TSent) / TRel))
            except:
                self.WordScore[word] = 0
        print("two in ",self.WordScore)
    # (Step 2) Feature extraction & (Step 3) Term score
    def start_or_end_with_stop_word(self,sentence):
        listeOFword=word_tokenize(sentence)
        if (listeOFword[0] in self.stopWord) or ( listeOFword[-1] in self.stopWord ):
            return True
        return False

    def __existe(self,word):
        for items in self.chunckDict:
            if word in items and items[word] in ['p', 'U', 'a']:
                return True
        return False
    def ngrams_generation(self,n=3):
        sentences = sent_tokenize(self.text)

        chunks = self.chunks(sentences)
        chunks = [ [ re.sub(r'[\W]+', ' ', tokens.lower()) for tokens in chunk] for chunk in chunks]

        chunks = [ [ tokens.lower() for tokens in chunk] for chunk in chunks]
        for tokens in chunks:
            for i in range(len(tokens)):
                cand=""
                if self.__existe(tokens[i]):
                    for j in range(n):
                        try:
                            if "." not in " ".join( tokens[i:i+j]) or "," not in " ".join( tokens[i:i+j]):
                                cand = " ".join( tokens[i:i+j] ) + " "
                                if (not self.start_or_end_with_stop_word(cand)) :
                                    self.candidateKeywords[cand].KF += 1
                        except:
                            pass
    def Proba(self,term1,term2):
        cooccure = 0
        occurance = 0
        for chunck in self.Chuncks:
            if term1 in chunck:
                occurance += 1
            if term1 in chunck and term2 in chunck and abs(chunck.index(term2) - chunck.index(term1)) == 1:
                cooccure += 1
        return cooccure / occurance

    def candidate_keyword_score(self):
        for candidats in   self.candidateKeywords.copy():
            tokens = candidats.split(" ")
            prod_S = 1
            sum_S = 0
            tokens.pop(-1)
            print( self.Proba(tokens[0],tokens[-1]),tokens[0],tokens[-1] )
            for i in  range(len(tokens)):
                prod_S = 1
                sum_S = 0
                if  tokens[i] in self.WordScore and tokens[i] not in self.stopWord :
                    prod_S *= self.WordScore[tokens[i]]
                    sum_S += self.WordScore[tokens[i]]
                else:
                    try:
                        probBefore = self.Proba(tokens[i],tokens[i-1])
                        probAfter = self.Proba(tokens[i], tokens[i + 1])
                        print("token:",tokens[i-1],tokens[i],tokens[i+1],probBefore,probAfter)
                        BigramProbability = probBefore * probAfter
                    except:
                        BigramProbability=0
                    prod_S *= 1 + (1 - BigramProbability)
                    sum_S -= (1 - BigramProbability)
                print(candidats,self.candidateKeywords[candidats].KF )
                try:
                    self.candidateKeywords[candidats].Score = prod_S / ( self.candidateKeywords[candidats].KF * (sum_S + 1) )
                except:
                    pass
        #self.candidateKeywords = sorted(self.candidateKeywords.keys(), key=lambda k: self.candidateKeywords[k].Score)

        sort = sorted(self.candidateKeywords.items(), key=lambda k: self.candidateKeywords[1].Score)

        for key, value in sort.copy():
            print("%s: %s" % (key, value.Score))

    def word_deduplication(self,threshold=0.5):
        keywords=[]
        #add first element  to liste
        for index,key in enumerate(self.candidateKeywords):
            if index>0:
                break
            print(key)
            keywords.append(key)
        for candidate in  self.candidateKeywords:
            skip = False
            for word in self.candidateKeywords:
                candidat=candidate
                candidat1=word
                if jaro_similarity(candidat,candidat1) > 0.3:
                    skip = True
                    break
            if not skip:
                keywords.append((candidat, candidat ))
        self.candidateKeywords = keywords

    def get_keyword(self):
        # (Step 1) Text pre-processing and candidate term identification
        self.preprocessing()
        # (Step 2) Feature extraction
        self.compute_term_statistics()

        # (Step 3) Term score
        self.term_score()
        # (Step 4) n-gram generation
        """
        self.ngrams_generation()
        # (Step 4) Candidate keyword score
        self.candidate_keyword_score()
        # (Step 5) Data deduplication
        self.word_deduplication(self)
        # (Step 5) Ranking
        keywords = sorted(self.candidateKeywords, key=lambda k: k[1])
        if len(keywords)>5:
            return keywords[:5]
        else:
            return keywords[:]
        """
text="""
We proposed an unsupervised keyphrase extraction model that incorporates the structural information and the semantic
information of a document. The structural information refers to the directed graph that is composed of keyphrase candidates
and topics. The weight between two candidates is computed by their relative distance in the document and the positions of
the corresponding sentences. Graph ranking algorithm is then applied to get the structural scores of the candidates. Then, the
semantic score is obtained by the similarity between candidate and all sentences. The final score of a candidate is the sum of
the structural score and the semantic score. The top N candidates with the highest scores are selected as the recommended
keyphrases. The comparison experiments on three widely used datasets show that our model achieves the best results in the
long documents and a competitive result in the short document. It indicates that our model is effective and is superior to the
state-of-the-art unsupervised models.
"""

pre = preprocessing(text)

pre.get_keyword()
from nltk.tokenize import sent_tokenize,word_tokenize
import re
class helper:
    def isAnacronimOrUpperCase(self, word):
        wordOfUppercases = 0
        for letter in word:
            if letter.isupper():
                wordOfUppercases += 1
        if wordOfUppercases == len(word):
            return 0
        if word[0].isupper() and wordOfUppercases == 1:
            return 1
        else:
            return 2
    def isMail(self,word):
        if re.match("([\w][_]?[.]?)*[@]([\w][.]?)+",word):
            return True
        return False
        # get the tage name of the word

    def getNameTag(self, word, pos):
        # return d if word is a number
        if re.match("^[\d+]+(.[\d]+)?$|^[\d+]+(,[\d]+)?$", word):
            return "d"
        # A term only formed by uppercase chars
        elif self.isAnacronimOrUpperCase(word) == 0:
            return "a"
        # return u if the word is either a mail or contains a number or symbols
        elif self.isMail(word) or re.search("(\w+)(\d+)(\w+)?|(\d+)(\w+)", word) or re.search("[#|:]+", word):
            return "u"
        # return U if the word is an acronim or an upper case word that does not in the beggining of the sentence
        elif self.isAnacronimOrUpperCase(word) == 1 and pos != 0:
            return "U"
        # return else otherwise
        else:
            return "p"
    def get_word_tags(self,dict,word,tag):
        compte=0
        for i in range(len(dict)):
            try:
                if(dict[i][word]== tag ):
                    compte+=1
            except:
                pass
        return compte

    def occurance(self,sentence,word1,word2):
        occur=False
        for i in range(len(sentence)):
            if word1 in sentence and word2 in sentence and abs( (sentence.index(word1)-sentence.index(word2)<=self.window)):
                return True
        return occur

    def getPosIndexSents(self,sentences,word):
        Sumpos=0
        for sentence in sentences:
            if word in sentence:
                Sumpos += sentences.index(sentence)
        return Sumpos
    def getSumIndexSents(self, sentences, word):
        Sumpos = 0
        for sentence in sentences:
            if word in sentence:
                Sumpos += sentences.index(sentence)
        return Sumpos
    def getPosIndexSents(self,sentences, word):
        Sumpos = []
        word=word.lower()
        for index, sentence in enumerate(sentences):
            if word in word_tokenize(sentence.lower()):
                Sumpos.append(index)
        return Sumpos


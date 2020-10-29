class Evaluator:
    """
        :param List: list of word of words get by the algorithme tested
        :param AnnontedList:
    """

    def __init__(self,List,AnnontedList):
        self.Lists = List
        self.AnnontedLists = AnnontedList
        self.retrivedInerRel = 0
        self.retrived=0
        self.Relevant=0
        self.getMesures()
    def getMesures(self):
        self.Relevant = len(self.AnnontedLists)
        self.retrived = len(self.Lists)
        for list,AnnontedList in zip(self.Lists,self.AnnontedLists):
            for word in list:
                for annot in AnnontedList:
                    if word == annot:
                        self.retrivedInerRel+=1
                        break
        print(self.Relevant,self.retrivedInerRel,self.retrived)
    def precesion(self):
        try:
            return self.retrivedInerRel/self.retrived
        except:
            return 0
    def rappel(self):
        try:
            return self.retrivedInerRel/self.Relevant
        except:
            return 0
    def F1_mesure(self,beta=0.5):
        beta *= beta
        rappelPre = self.rappel() * self.precesion()
        try:
            return ( (1+beta)*  rappelPre)  / ( (beta *self.precesion())+self.rappel() )
        except:
            return 0

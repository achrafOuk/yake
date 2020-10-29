import csv
class Reader:
    def __init__(self,file):
        self.file=file
        self.Sites =[]
        self.Annonted =[]
    def getSitesName(self):
        with open(self.file) as myfile:
            csv_reader = csv.reader(myfile, delimiter=';')
            for index,row in enumerate(csv_reader):
                if  index!=0:
                    self.Sites.append(row[0])
                    self.Annonted.append(row[1])
        return self.Sites,self.Annonted





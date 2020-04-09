import pandas as pd
import numpy as np
import math



class C45:
    def __init__(self,classes,attr,aVal,datas):
        self.data = datas
        self.classes = classes
        self.frequency = len(self.classes)
        self.numAttr = len(attr) 
        self.attrValues = aVal
        self.attr = attr
        self.tr = None

    def generateTree(self):
        self.tr = self.treeGenerate(self.data, self.attr)

    def treeGenerate(self, cData, cAttr):
        allSame = cData[0][-1]
        for r in cData:
            if r[-1] != cData[0][-1]:
                allSame = False

        if len(cData) == 0:
            return Node(True, "Fail", None)
        elif allSame != False:
            return Node(True, allSame, None)
        elif len(cAttr) == 0:
            frequency = [0]*self.frequency
            for r in cData:
                ind = self.classes.index(r[-1])
                frequency[ind] +=1
            maxInd = frequency.index(max(frequency))
            mClass = self.classes[maxInd]
            return Node(True, mClass, None)
        else:
            (highAttr,highHeru,split) = self.splitAttr(cData, cAttr)
            remainingattr = cAttr[:]
            remainingattr.remove(highAttr)
            node = Node(False, highAttr, highHeru)
            node.child = [self.treeGenerate(sub, remainingattr) for sub in split]
            return node

    def isNumericCont(self, attr):
        if attr not in self.attr:
            print("there is no attribute equals to "+ attr+" in the dataset")
            exit
        elif len(self.attrValues[attr]) == 1 and self.attrValues[attr][0] == "cont":
            return True
        else:
            return False

    def splitAttr(self, cData, cAttr):
        maxEnt = -1*float("inf")
        highAttr = -1
        split = []
        #None for discrete attr, herustic value for cont attr
        highHeru = None
        for attr in cAttr:
            attrIndex = self.attr.index(attr)
            # if the data is continuous numerical value
            # sort the data inorder and calculate and choose highest information gain from all adjacent pairs
            if self.isNumericCont(attr):
                cData.sort(key = lambda x: x[attrIndex])
                for j in range(0, len(cData) - 1):
                    if cData[j][attrIndex] != cData[j+1][attrIndex]:
                        herustic = round((cData[j][attrIndex] + cData[j+1][attrIndex]) / 2,5)
                        lessThan = []
                        greatThan = []
                        for r in cData:
                            if(r[attrIndex] > herustic):
                                greatThan.append(r)
                            else:
                                lessThan.append(r)
                        tmp = self.infoGain(cData, [lessThan, greatThan])
                        if tmp >= maxEnt:
                            split = [lessThan, greatThan]
                            maxEnt = tmp
                            highAttr = attr
                            highHeru = herustic
            #if the data is categorical
            #split the data into number of unique categories of the attr and choose one with the highest information gain
            else:
                valueattr = self.attrValues[attr]
                sub = [[] for a in valueattr]
                for r in cData:
                    for index in range(len(valueattr)):
                        if r[i] == valueattr[index]:
                            sub[index].append(r)
                            break
                tmp = infoGain(cData, sub)
                if tmp > maxEnt:
                    maxEnt = tmp
                    split = sub
                    highAttr = attr
                    highHeru = None
        return (highAttr,highHeru,split)


    #calculate the info gain of the current datas in a node 
    def infoGain(self,cData, sub):
        S = len(cData)
        impurityBeforeSplit = self.getEntropy(cData)
        weights = [len(sub)/S for sub in sub]
        impurityAfterSplit = 0
        for i in range(len(sub)):
            impurityAfterSplit += weights[i]*self.getEntropy(sub[i])
        totalinfoGain = impurityBeforeSplit - impurityAfterSplit
        return totalinfoGain

    #calculate the entropy from the dataset
    def getEntropy(self, ds):
        S = len(ds)
        entropy = 0
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for r in ds:
            classIndex = list(self.classes).index(r[-1])
            num_classes[classIndex] += 1
        num_classes = [x/S for x in num_classes]
        for num in num_classes:
            if num != 0:
                entropy += num*math.log(num,2)
        entropy*=-1
        return entropy


    def displayDT(self):
        self.displayNode(self.tr)

    def displayNode(self, node, ph=""):
        if not node.isLeaf:
            #if the value is catagorical
            if node.herustic is None:
                for index,child in enumerate(node.child):
                    if child.isLeaf:
                        print(ph + node.label + " = " + attr[index] + "  -->  " + child.label)
                    else:
                        print(ph + node.label + " = " + attr[index] + "  -->  ")
                        self.displayNode(child, ph + "    ")
            #if the value is numerical continous
            else:
                leftChild = node.child[0]
                rightChild = node.child[1]
                if leftChild.isLeaf:
                    print(ph + node.label + " <= " + str(round(node.herustic,5)) + "  -->  " + leftChild.label)
                else:
                    print(ph + node.label + " <= " + str(round(node.herustic,5))+"  -->  ")
                    self.displayNode(leftChild, ph + "    ")

                if rightChild.isLeaf:
                    print(ph + node.label + " > " + str(round(node.herustic,5)) + "  -->  " + rightChild.label)
                else:
                    print(ph + node.label + " > " + str(round(node.herustic,5)) + "  -->  ")
                    self.displayNode(rightChild , ph + "    ")
            

    def testData(self, data,total):
        cor = 0
        for index, row in data.iterrows():
            res =  self.testNode(self.tr,row)
            if res:
                cor+=1
        print("SUCCESS = " + str(cor)+"/"+str(total)+" test cases")
        accuracy = cor/total
        print("accuracy = " + str(round(accuracy*100,2))+"%")
    
    def testNode(self,node,data):
        dVal = round(data[self.getColNum(node.label)],5)
        if not node.isLeaf:
            if node.herustic is None:
                print("WIP")
            else:
                leftChild = node.child[0]
                rightChild = node.child[1]
                if dVal <= round(node.herustic,5):
                    if leftChild.isLeaf:
                        # print(leftChild.label,end=" = ")
                        # print(data[11])
                        
                        if str(leftChild.label) == str(data[11]):
                            return True
                        else:
                            return False
                        
                        # return str(leftChild.label) == str(data[11])
                    else:
                        return self.testNode(leftChild,data)
                else:
                    if rightChild.isLeaf:
                        # print(rightChild.label,end=" = ")
                        # print(data[11])
                        if str(rightChild.label) == str(data[11]):
                            return True
                        else:
                            return False
                        # return str(rightChild.label) == str(data[11])
                    else:
                        return self.testNode(rightChild,data)

    def getColNum(self, label):
        if label == "fixedAcidity":
            return 0
        elif label == "volatileAcidity":
            return 1
        elif label == "citricAcid":
            return 2
        elif label == "residualSugar":
            return 3
        elif label == "chlorides":
            return 4
        elif label == "freeSulfurDioxide":
            return 5
        elif label == "totalSulfurDioxide":
            return 6
        elif label == "density":
            return 7
        elif label == "pH":
            return 8
        elif label == "sulphates":
            return 9
        elif label == "alcohol":
            return 10
        else:
            return 11


class Node:
    def __init__(self, isLeaf, label, herustic):
        self.label = label
        self.isLeaf = isLeaf
        self.herustic = herustic
        self.child = []
    
    def display(self):
        print(self.label)
        print(self.herustic)
        print(self.isLeaf)
        for i in self.child:
            print(i.label)
        

#start with assumption that the target of the tree is the last column of the dataset
def preProcess(df):
    # df = pd.read_csv(fileName)
    # print(df[0])
    # for index, row in df.iterrows():
    #     print(row)
    #     print(row[11])
    #     # for i in range(len(row)):
    #     #     print(row[i])
    #     break


    tmp=df.dtypes
    # print(len(tmp))
    cates = [[0 for x in range(2)] for y in range(len(tmp)-1)] 
    ind = 0
    for i in tmp:
        if ind == len(tmp)-1:
            break
        else:
            if i=="float64":
                cates[ind][1]= "cont"
                ind+=1
            else:
                cates[ind][1]="categorical"
                ind+=1
    ind=0
    for i in df:
        if ind == len(tmp)-1:    
            break
        else:
            cates[ind][0]=i
            ind+=1
    classes=[]
    for i in df.iloc[:,-1].unique():
        classes.append(i)
    classes = classes
    attr = []
    aVal={}
    for i in cates:
        attr.append(i[0])
    for i in cates:
        aVal[i[0]] = [i[1]]
    datas = []
    ind = 0
    for index, r in df.iterrows():
        temp=[]
        for i in r:
            temp.append(i)
        datas.append(temp)
    return datas , aVal, attr, classes

def dataSplit(fileName):
    df = pd.read_csv(fileName)
    # df = df.round(5)
    df["quality"] = df["quality"].astype(str)
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= 0.7
 
    train = df[msk]
    test = df[~msk]

    # df = df.iloc[:, :-1]
    train = train.iloc[:,:-1]
    test = test.iloc[:,:-1]
    df = df.iloc[:,:-1]
    return train, test,df



def dataSplit2(fileName):
    df = pd.read_csv(fileName)
    # df = df.round(5)
    df["quality"] = df["quality"].astype(str)
    rng = np.random.RandomState()

    train = df.sample(frac=0.7, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]
    return train, test,df


def main():
    train, test ,whole= dataSplit2("winequality-white.csv")
    # print(whole["quality"].unique())
    while len(train["quality"].unique()) != len(test["quality"].unique()):
        train, test,whole = dataSplit2("winequality-white.csv")
    
    # print(len(test))
    # print("HERE")
    whole = whole.round(5)
    datas, aVal, attr,classes=preProcess(whole)
    # print(classes)
    # print(attr)
    # print(aVal)
    # print(datas)
    c45 = C45(classes,attr,aVal,datas)
    # for i in c45.data:
    #     print(i)
    c45.generateTree()

    # c45.tr.display()
    #TEST THE VALIDITY
    print("___________TEST VALIDITY___________")
    c45.testData(test,len(test))
    print("\n___________TREE___________\n")
    c45.displayDT()
 
    


if __name__=="__main__":
    for i in range(1,100):
        try:
            main()
        except Exception as e:
            continue
        else:
            break


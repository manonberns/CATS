import sys
import numpy as np

# Initialising variables
arrayCGHData = ""
arrayGroups = ""
cghData = []
groupDataDict = {}
compiledData = []
cghDataList = []
cghDataDict = {}
array_data = np.array([0,0])

try:
    arrayCGHData = sys.argv[1]
    arrayGroups = sys.argv[2]
except IndexError:
    print("No data files given. I will use the default files!!")
    arrayCGHData = "Train_call.txt"
    arrayGroups = "Train_Clinical.txt"

fileData = open(arrayCGHData, "r")
fileGroups = open(arrayGroups, "r")

# Get the data from CGH Data file
for line in fileData:
	lineTerm = line.rstrip("\n").split("\t")
	cghData.append(lineTerm)	   
fileData.close()
cghDataMatrix = np.matrix(cghData)
cghDataMatrixTrans = cghDataMatrix.transpose()


# Get the data from the clinical data file
for line in fileGroups:
    lineTerm = line.rstrip("\n").split("\t")
    groupDataDict[lineTerm[0]] = lineTerm[1]
fileGroups.close()


compiledMatrix=np.insert(cghDataMatrixTrans, 1, "-", axis=1 )

for d in range(0,len(cghDataMatrixTrans)):
	if str(cghDataMatrixTrans[d,0]) in groupDataDict.keys():
		compiledMatrix[d,1]=groupDataDict[cghDataMatrixTrans[d,0]]
	
print (compiledMatrix)
	

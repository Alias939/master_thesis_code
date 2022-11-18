import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Opening JSON file
testing = open("C:\\Users\\trekk\\Desktop\\thesis\\ScenarioCorpusMatching-master\\data\\ai2MatchesGSETesting.json")
validation = open("C:\\Users\\trekk\\Desktop\\thesis\\ScenarioCorpusMatching-master\\data\\ai2MatchesGSEValidation.json")
training = open("C:\\Users\\trekk\\Desktop\\thesis\\ScenarioCorpusMatching-master\\data\\ai2MatchesGSETraining.json")

testingData = json.load(testing)
trainingData = json.load(training)
validationData = json.load(validation)

import matplotlib.pyplot as plt

# possible configurations:
# [trainingData,validationData] - chosing threshold on both training and val
# [trainingData] - chosing threshold only on training
# [testingData] - testing chosen threshold (change for loop)

# config = [trainingData,validationData]
#config = [testingData]
config = [trainingData]

# models = ['clips/mfaq','sentence-transformers/multi-qa-mpnet-base-dot-v1','all-mpnet-base-v2','all-distilroberta-v1','all-MiniLM-L12-v2']
# models = ['sentence-transformers/multi-qa-mpnet-base-dot-v1']
# models = ['all-mpnet-base-v2','all-distilroberta-v1','all-MiniLM-L12-v2']

# models = ['all-MiniLM-L12-v2']

models = ['clips/mfaq']

totalMatched = 0
totalUnmatched = 0

if config == [trainingData, validationData]:
    totalMatched = 424
    totalUnmatched = 326
if config == [trainingData]:
    totalMatched = 306
    totalUnmatched = 227
if config == [testingData]:
    totalMatched = 147
    totalUnmatched = 77

print(totalMatched)

for modeltype in models:

    xvalue = []
    mn = []
    umn = []
    fp = []
    fn = []
    flag = 0

    model = SentenceTransformer(modeltype)

    for thresh in range(0, 100, 1):

        print(thresh)
        thresh = thresh / 100
        xvalue.append(thresh)

        index = 0;
        goldenMatch = []
        modelMatch = []
        matchesNumber = 0
        unmatchesNumber = 0
        falseNegative = 0
        falsePositive = 0
        count = 0

        for configuration in range(len(config)):
            ########
            # Data
            ########

            for index in range(0, len(config[configuration])):
                testingSentences = []
                modelScore = []

                testingSentences.append('<Q>' + config[configuration][index]['input']['text'])

                for i in config[configuration][index]['options']:
                    testingSentences.append('<A>' + i['text'])

                goldenMatch = config[configuration][index]['goldenMatch']
                embeddings = model.encode(testingSentences, convert_to_tensor=True)

                # print(len(embeddings))
                for j in range(len(embeddings) - 1):
                    modelScore.append(
                        cosine_similarity(embeddings[0].reshape(1, -1), embeddings[j + 1].reshape(1, -1))[0][0])

                    # if the biggest scoring answer is higher than the threshold, if the answer is the same as the goldenmatch, increase matched result
                if (max(modelScore)) > (thresh):
                    if ((modelScore.index(max(modelScore)) + 1 is goldenMatch)):
                        matchesNumber += 1
                    else:
                        falsePositive += 1
                # if none of the answers are above the threshold, and the golden match is None (unmatched), unmatched result is increased
                else:
                    if (goldenMatch is None):
                        unmatchesNumber += 1
                    else:
                        falseNegative += 1

        fp.append(falsePositive)
        fn.append(falseNegative)
        mn.append(matchesNumber)
        umn.append(unmatchesNumber)

        # if(len(mn)>=2):
        # if (abs(matchesNumber/totalMatched - unmatchesNumber/totalUnmatched) > abs(mn[-2]/totalMatched - umn[-2]/totalUnmatched)):
        # flag = 1
        # break

    normmn = [x / totalMatched for x in mn]
    normumn = [x / totalUnmatched for x in umn]

    print(modeltype)

    plt.plot(xvalue, normmn, label='Matched')
    plt.plot(xvalue, normumn, label='Unmatched')
    plt.legend()
    plt.ylabel('Percentage of matches %')
    plt.xlabel('Threshold')
    plt.title("Normalized threshold comparison for " + modeltype + " model ")
    plt.grid()
    plt.show()

    for i in range(0, len(normmn)):
        f1 = '%.3f' % ((2 * mn[i]) / (2 * mn[i] + fp[i] + fn[i]))
        f2 = '%.3f' % (mn[i] / (mn[i] + ((totalMatched + totalUnmatched) - (mn[i] + umn[i])) * 1 / 2))
        print(mn[i], ' ', umn[i], ' ', normmn[i] - normumn[i], ' ', xvalue[i], f1, f2)

    with open('test_file.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Threshold","True Positives", "True Negatives", "False Positives", "False Negatives"])
        for i in range(0, len(mn)):
            writer.writerow([xvalue[i],mn[i], umn[i], fp[i], fn[i]])

testing.close()
training.close()
validation.close()
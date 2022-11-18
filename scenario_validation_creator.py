from sentence_transformers import SentenceTransformer, util, models
from sklearn.metrics.pairwise import cosine_similarity
import torch
import csv
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



def validate_scenario(final_model,final_tokenizer):
    print("run model")

    import json

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


    #############   VALIDATION DATASET

    # config = [trainingData,validationData]
    config = [trainingData]
    # config = [testingData]

    # models = ['clips/mfaq','sentence-transformers/multi-qa-mpnet-base-dot-v1','all-mpnet-base-v2','all-distilroberta-v1','all-MiniLM-L12-v2']
    # models = ['sentence-transformers/multi-qa-mpnet-base-dot-v1']
    # models = ['all-mpnet-base-v2','all-distilroberta-v1','all-MiniLM-L12-v2']

    # models = ['all-MiniLM-L12-v2']

    #models = ['clips/mfaq']
    #models =[final_model]

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

    xvalue = []
    mn = []
    umn = []
    fp = []
    fn = []

    model = final_model

    #go through all of the threshold values we want

    for thresh in range(72,81, 1):

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

        #for each model
        for configuration in range(len(config)):
            ########
            # Data
            ########

            for index in range(0, len(config[configuration])):
                testingSentences = []
                modelScore = []

                #append the question to the block
                testingSentences.append('<Q>' + config[configuration][index]['input']['text'])
                #append all of the answers to the block

                for i in config[configuration][index]['options']:
                    testingSentences.append('<A>' + i['text'])


                #get the goldenmatch value of the block
                goldenMatch = config[configuration][index]['goldenMatch']

                #embeddings = model.encode(testingSentences, convert_to_tensor=True)
                encoded_input = final_tokenizer(testingSentences, padding=True, truncation=True, max_length=256, return_tensors='pt')

                with torch.no_grad():
                    model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


                #get the scores for each answer
                for j in range(len(sentence_embeddings) - 1):
                    modelScore.append(
                        cosine_similarity(sentence_embeddings [0].reshape(1, -1), sentence_embeddings [j + 1].reshape(1, -1))[0][0])


                #if the biggest scoring answer is higher than the threshold, if the answer is the same as the goldenmatch, increase matched result
                if (max(modelScore)) > (thresh):
                    if ((modelScore.index(max(modelScore)) + 1 is goldenMatch)):
                        matchesNumber += 1
                    else:
                        falsePositive += 1
                #if none of the answers are above the threshold, and the golden match is None (unmatched), unmatched result is increased
                else:
                    if (goldenMatch is None):
                        unmatchesNumber += 1
                    else:
                        falseNegative +=1
                #matchedNumber = true positive
                #unmatchesNumber = true negative


        #Plotting

        mn.append(matchesNumber)
        umn.append(unmatchesNumber)
        fp.append(falsePositive)
        fn.append(falseNegative)

        # if(len(mn)>=2):
        #     if (abs(matchesNumber/totalMatched - unmatchesNumber/totalUnmatched) > abs(mn[-2]/totalMatched - umn[-2]/totalUnmatched)):
        #         flag = 1
        #         break

    normmn = [(x / totalMatched)*100 for x in mn]
    normumn = [(x / totalUnmatched)*100 for x in umn]

    print("clips/mfaq")

    plt.plot(xvalue, normmn, label='Matched')
    plt.plot(xvalue, normumn, label='Unmatched')
    plt.legend()
    plt.ylabel('Percentage of matches %')
    plt.xlabel('Threshold')
    plt.title("Normalized threshold comparison for " + "clips/mfaq" + " model ")
    plt.grid()
    plt.show()

    for i in range(0, len(normmn)):
        f1 = '%.3f' % (mn[i] / (mn[i] + ((totalMatched + totalUnmatched) - (mn[i] + umn[i])) * 1 / 2))
        #print(mn[i], ' ', umn[i], ' ', normmn[i] - normumn[i], ' ', xvalue[i], f1)
        print(mn[i], ' ', umn[i], ' ', normmn[i] - normumn[i], ' ', xvalue[i], f1)

    with open('test_file.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Threshold","True Positives", "True Negatives", "False Positives", "False Negatives"])
        for i in range(0, len(mn)):
            writer.writerow([xvalue[i],mn[i], umn[i], fp[i], fn[i]])

    testing.close()
    training.close()
    validation.close()


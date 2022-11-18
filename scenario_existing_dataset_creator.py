from datasets import Dataset
import csv


def create_existing_scenario_dataset():
    final_dict = {'id': [],
                  'language': [],
                  'num_pairs': [],
                  'domain': [],
                  'qa_pairs': []}

    import json

    # Opening JSON file
    testing = open("C:\\Users\\trekk\\Desktop\\thesis\\ScenarioCorpusMatching-master\\data\\ai2MatchesGSETesting.json")
    validation = open("C:\\Users\\trekk\\Desktop\\thesis\\ScenarioCorpusMatching-master\\data\\ai2MatchesGSEValidation.json")
    training = open("C:\\Users\\trekk\\Desktop\\thesis\\ScenarioCorpusMatching-master\\data\\ai2MatchesGSETraining.json")



    testingData = json.load(testing)
    trainingData = json.load(training)
    validationData = json.load(validation)



    ########     FINETUNING DATASET

    #config = [trainingData]
    config = [testingData]
    #config = [validationData]

    for index in range(0, len(config[0])):



        testingSentences = ''
        goldenMatch = config[0][index]['goldenMatch']

        if goldenMatch is None:
            #testingSentences = '<A>None'
            testingSentences = None
        else:
            testingSentences = '<A>' + config[0][index]['options'][goldenMatch-1]['text']

        block = {'question': '<Q>' + config[0][index]['input']['text'],
                 'answer': testingSentences,
                 'language': 'nl',
                 'id':index,
                 'page_id':index
                 }




        final_dict['qa_pairs'].append(block)


        final_dict['id'].append(index)
        final_dict['language'].append('nl')
        final_dict['num_pairs'].append(1)
        final_dict['domain'].append("training")

    dataset = Dataset.from_dict(final_dict)
    return dataset

# tp = [1,2]
# tn = [2,4]
# fp = [6,6]
# fn = [2,5]
#
# with open('test_file.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["True Positives", "True Negatives", "False Positives", "False Negatives"])
#     for i in range(0, len(tp)):
#         writer.writerow([tp[i],tn[i],fp[i],fn[i]])


# dataset = create_existing_scenario_dataset()
# print(dataset['num_pairs'])
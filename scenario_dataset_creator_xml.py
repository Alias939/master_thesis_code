# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import csv
import xmltodict

def create_scenario_dataset_xml():
    scenario_name = "SamenwerkenOT_correct.xml"

    with open(scenario_name, 'r') as f:
        data = f.read()
    players = {}
    computers = []
    situations = []

    playerinput = []
    computerinput = []
    tempid = []


    final_dict = {'id': [],
               'language': [],
               'num_pairs': [],
               'domain': [],
               'qa_pairs': []}




    my_dict = xmltodict.parse(data)

    for interleave in my_dict['scenario']['sequence']['interleave']:
        for computerstatement in interleave['dialogue']['statements']['computerStatement']:
            if computerstatement['responses'] is None:
                continue
            else:
                id = []
                if isinstance(computerstatement['responses']['response'], list):
                    for response in computerstatement['responses']['response']:
                        id.append(response['@idref'])
                else:

                    id.append(computerstatement['responses']['response']['@idref'])
                temp = []
                temp.append(computerstatement['text']['#text'])
                temp.append(id)
                computers.append(temp)
            # print(' ')

    for interleave in my_dict['scenario']['sequence']['interleave']:
        for playerstatement in interleave['dialogue']['statements']['playerStatement']:
            players[playerstatement['@id']] = playerstatement['text']['#text']

    for interleave in my_dict['scenario']['sequence']['interleave']:
        if 'situationStatement' in interleave['dialogue']['statements']:
            situations.append(interleave['dialogue']['statements']['situationStatement']['@id'])

    test = 0

    for situation in situations:
        for test in range(len(computers)):
            if situation in computers[test][1]:
                computers[test][1].remove(situation)

    for j in computers:
        if len(j[1]) == 0:
            computers.remove(j)

    for test in range(len(computers)):
        answer = ''
        for i in computers[test][1]:
            answer = players[i] +'\n\n'+ answer
        answer = answer[:-2]

        block = {'question': '<Q>' + computers[test][0],
                 'answer': 'A' + answer,
                 'language': 'nl',
                 'id': test,
                 'page_id':test}

        final_dict['qa_pairs'].append(block)
        tempid.append(test)

        final_dict['id'].append(test)
        final_dict['language'].append('nl')
        final_dict['num_pairs'].append(len(computers))
        final_dict['domain'].append(scenario_name)

    from datasets import Dataset
    dataset = Dataset.from_dict(final_dict)
    return dataset

#dataset = create_scenario_dataset_xml()
#print(dataset['id'])
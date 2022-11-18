# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import csv
def create_scenario_dataset():
    playerinput = []
    computerinput = []
    temp = []
    flag = 0
    my_dict = {'id':[],
               'language':[],
               'num_pairs':[],
               'domain':[],
               'qa_pairs':[[]]}

    scenario_name = 'test_scenario.csv'
    #scenario_name = 'dutch_scenario_test.csv'

    with open(scenario_name, mode='r') as infile:
        reader = csv.reader(infile)

        next(reader)

        for row in reader:

            if row[0] == 'Computer':
                if temp:
                    temp = []
                    playerinput.append(temp)
                computerinput.append(row[1])
            if row[0] == 'Player':
                temp.append(row[1])


    print(len(playerinput))
    print(len(computerinput))


    #for i in range(len(computerinput)):
    my_dict['id'].append(0)
    my_dict['language'].append('nl')
    my_dict['num_pairs'].append(len(computerinput))
    my_dict['domain'].append(scenario_name)


    for k in range(len(computerinput)):
        answer = ''
        if len(playerinput[k]) == 1:
            answer = str(playerinput[k][0])
        else:
            for j in playerinput[k]:
                answer = j + '\n\n' + answer
            answer = answer[:-2]

        block = {'question': 'Q' + computerinput[k],
                 'answer':'<A>' + answer,
                 'language': 'nl'}

        my_dict['qa_pairs'][0].append(block)

    print(my_dict)

    # for k in range(len(computerinput)):
    #     my_dict['id'].append(k)
    #     my_dict['language'].append('nl')
    #     my_dict['num_pairs'].append(len(computerinput))
    #     my_dict['domain'].append(scenario_name)
    #
    #
    #     answer = ''
    #
    #     if len(playerinput[k]) == 1:
    #         answer = str(playerinput[k][0])
    #     else:
    #         for j in playerinput[k]:
    #             answer = j + '\n\n' + answer
    #         answer = answer[:-2]
    #
    #
    #
    #     block = {'question': computerinput[k],
    #              'answer':answer,
    #              'language': 'nl'}
    #
    #     my_dict['qa_pairs'].append(block)
    #
    # print(my_dict['qa_pairs'][4])
    #
    # #print(computerinput[k])
    # #print(playerinput[k])
    #
    from datasets import Dataset
    dataset = Dataset.from_dict(my_dict)

    return dataset

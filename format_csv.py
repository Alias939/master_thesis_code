import csv


training_file = 'C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\DEBUG\\training1.csv'
testing_file = 'C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\DEBUG\\testing1.csv'
testing_training_file = 'C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\DEBUG\\testing-training1.csv'
training_testing_file = 'C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\DEBUG\\training-testing1.csv'


final_file = 'C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\DEBUG\\final.csv'

def format_final():
    with open(final_file, 'w',newline='') as csvfile:
        finalwriter = csv.writer(csvfile)

        with open(testing_file,'r') as testing_csv:
            with open(testing_training_file, 'r') as testing_training_csv:
                with open(training_file, 'r') as training_csv:
                    with open(training_testing_file, 'r') as training_testing_csv:

                        training_reader = csv.reader(training_csv)
                        testing_training_reader = csv.reader(testing_training_csv)
                        testing_reader = csv.reader(testing_csv)
                        training_testing_reader = csv.reader(training_testing_csv)

                        for skip in range(50):
                            training_row = next(training_reader)
                            testing_training_row = next(testing_training_reader)
                            testing_row = next(testing_reader)
                            training_testing_row = next(training_testing_reader)

                        finalwriter.writerow(["Threshold", "True Positives",'','','', "True Negatives",'','','',"False Positives",'','','',"False Negatives",'','',''])
                        finalwriter.writerow(
                            ["", "Testing", 'Finetuned on training', 'Training', 'Finetuned on testing'])

                        for i in range(50):

                            training_row = next(training_reader)
                            testing_training_row = next(testing_training_reader)
                            testing_row = next(testing_reader)
                            training_testing_row = next(training_testing_reader)


                            finalwriter.writerow([training_row[0],
                                                  testing_row[1],training_testing_row[1],training_row[1],testing_training_row[1],
                                                  testing_row[2],training_testing_row[2],training_row[2],testing_training_row[2],
                                                  testing_row[3],training_testing_row[3],training_row[3],testing_training_row[3],
                                                  testing_row[4],training_testing_row[4],training_row[4],testing_training_row[4]])



def format_single():
    with open('C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\stochastic\\0.65 - 0.75\\') as input, open('C:\\Users\\trekk\\Desktop\\thesis\\Thesis Andrei\\Results\\DEBUG\\training1.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        for row in csv.reader(input):
            if any(field.strip() for field in row):
                writer.writerow(row)
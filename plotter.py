import matplotlib.pyplot as plt

totalMatched = 147
totalUnmatched = 77

mn = []
umn = []

normmn = [x / totalMatched for x in mn]
normumn = [x / totalUnmatched for x in umn]

xvalue = []

for i in range(0,100,1):
    thresh = thresh / 100
    xvalue.append(thresh)

plt.plot(xvalue, normmn, label='Matched')
plt.plot(xvalue, normumn, label='Unmatched')
plt.legend()
plt.ylabel('Percentage of matches %')
plt.xlabel('Threshold')
plt.title("Normalized threshold comparison for " + "clips/mfaq" + " model ")
plt.grid()
plt.show()
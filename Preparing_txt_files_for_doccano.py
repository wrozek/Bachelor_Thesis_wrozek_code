import pandas as pd

#take the datasets with lemmatized and tokenized columns
data = pd.read_csv('Test_Data_refugee_all_punct.tsv', sep='\t', header = 0)


#Remove double spaces and create files with the content of each article
file1 = 'BBBBBBBBBBBBBBBBBBBBBdoccanofile{}.txt'
possible_data= []
n = 0 # to number the files
for i in range(len(data)):
    if len(data.iloc[i][0]) < 1000 and len(data.iloc[i][0]) > 700:
        new_str = ""
        s = data.iloc[i][0]
        for idx, sign in enumerate(s):
            if sign == " " and s[idx + 1] == " ":
                new_str += ""
            else:
                new_str += sign
        print(new_str)
        with open(r''+file1.format(n)+'.txt', "w", encoding="utf-8") as file:
            file.write(new_str)
        new_str.to_csv(r''+file.format(n)+'.txt', header=None, index=None, sep='\t', mode='a')

    n += 1

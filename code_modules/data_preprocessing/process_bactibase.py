from code_modules.data_preprocessing.preprocessing_functions import \
    test_sequence_column, fasta2csv

data = fasta2csv(path='data/bactibase/BACTIBASE021419.txt')
test_sequence_column(in_data=data)

data.to_csv("data/bactibase/bactibase021419.csv", sep='\t')

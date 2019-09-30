% please input the train positive and negative fasta file path and also
% test fasta file path to get the result of prediction 1(Anti-microbial
% peptide)or 0(Non-antimicrobial Peptide), and their corresponding score
% which should be in the closed interval [0,1] 
% all the resulr will be evaluated by random forest with 100 trees.
function [predict_result] = main_function(test_fasta)
    % train_positive_fasta should be your positive train fasta file path
    % train_negative_fasta should be your negative train fasta file path
    train_positive_fasta = 'trian_po_set3298_for_ampep_sever.fasta';
    train_negative_fasta = 'trian_ne_set9894_for_ampep_sever.fasta';
    % generate the train data by the input positive and negative fasta file
    [train_data,train_rownames,train_colnames] = ...
        train_seqs2distribution(train_positive_fasta,train_negative_fasta);
    % generate the test data by the input test fasta file
    [test_data,test_rownames,test_colnames] = test_seqs2distribution(test_fasta);
    [prediction,score]=RF_test_yan(train_data, test_data);
    predict_result=table(prediction,score,'rownames',test_rownames);

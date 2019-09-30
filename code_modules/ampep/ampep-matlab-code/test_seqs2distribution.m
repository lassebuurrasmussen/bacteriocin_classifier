% generate the feature matrix from all the input sequences 
% and add the class vector to the end column of the feature matrix
function [d_mat,rownames,colnames] = test_seqs2distribution(test_fasta_file)
% read positive and negative sequences from fasta file
    [test_name,test_seq] = fastaread(test_fasta_file);
    test_len = length(test_seq);
    d_mat=[];
% generate distribution feature from all sequences
    for i=1:test_len
        seq=test_seq{i};
        [d_vector,ft_names] = Seq2Distribution(seq);
        d_mat=[d_mat;d_vector];
    end
    rownames = test_name;
    colnames = ft_names;
% generate the feature matrix from all the input sequences 
% and add the class vector to the end column of the feature matrix
function [d_mat,rownames,colnames] = train_seqs2distribution...
    (positive_fasta_file,negative_fasta_file)
% read positive and negative sequences from fasta file
    [po_name,po_seq] = fastaread(positive_fasta_file);
    [ne_name,ne_seq] = fastaread(negative_fasta_file);
    po_len = length(po_seq);
    ne_len = length(ne_seq);
% generate class vector
    class = [ones(po_len,1);zeros(ne_len,1)];
    all_seq = [po_seq,ne_seq];
    len = length(all_seq);
    d_mat=[];
% generate distribution feature from all sequences
    for i=1:len
        seq=all_seq{i};
        [d_vector,ft_names] = Seq2Distribution(seq);
        d_mat=[d_mat;d_vector];
    end
% add class to the end column of feature matix
    d_mat = [d_mat,class];
    rownames = [po_name,ne_name];
    colnames = [ft_names,{'class'}];
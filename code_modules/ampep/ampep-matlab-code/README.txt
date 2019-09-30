=======================================================================
MATLAB code and datasets for AmPEP

Developed by:
Yan Jielu yb87410@connect.umac.mo
Computational Biology and Bioinformatics Lab (CBBio)                        
University of Macau 

Reference:
Pratiti Bhadra, Jielu Yan, Jinyan Li, Simon Fong and Shirley W. I. Siu
AmPEP: Sequence-based prediction of antimicrobial peptides using 
distribution patterns of amino acid properties and random forest
Scientific Reports 8, 1697 (2018) 
                                                                       
Visit http://cbbio.cis.umac.mo for more information.                                               
=======================================================================

Required software:
----------------------------------------
MATLAB R2018a or above 
Bioinfomatics Toolbox 
Statistics and Machine Learning Toolbox

Brief Instruction:
----------------------------------------
If you want to predict your peptide sequence to identify it as Antimicrobial peptide(AMP) or 
Non-antimicrobial Peptide (Non-AMP) by our AmPEP code, please implement this command in your matlab command window: 
[predict_result] = main_function(test_fasta_path);


test_fasta should be your test fasta file path

predict_result is a table that contains the fasta name of all test sequences 
and prediction result which can be 1(Anti-microbial peptide)or 0(Non-antimicrobial Peptide)
and score which should be in the closed interval [0,1].

Prediction is performed by random forest with 100 trees.


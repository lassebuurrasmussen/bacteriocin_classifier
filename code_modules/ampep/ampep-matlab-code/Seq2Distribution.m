function [d_vector,ft_names] = Seq2Distribution(seq)
% 7 properties and their three groups cited the paper of 
% Composition, Transition and Distribution (CTD) -
% A Dynamic Feature For Predictions Based on
% Hierarchical Structure Of Cellular Sorting
    charge = [{'ACFGHILMNPQSTVWY'},{'DE'},{'KR'}];
    hydrophobicity = [{'CFILMVW'},{'AGHPSTY'},{'DEKNQR'}];
    normalized_vander_waals = [{'ACDGPST'},{'EILNQV'},{'FHKMRWY'}];
    polarity = [{'CFILMVWY'},{'AGPST'},{'DEHKNQR'}];
    polarizability = [{'ADGST'},{'CEILNPQV'},{'FHKMRWY'}];
    secondary_structure = [{'DGNPS'},{'AEHKLMQR'},{'CFITVWY'}];
    solvent_accessibility = [{'ACFGILVW'},{'HMPSTY'},{'DEKNRQ'}];
    property_group = {charge, hydrophobicity, normalized_vander_waals, ...
        polarity, polarizability, secondary_structure, ...
        solvent_accessibility};
    property_name = {'charge', 'hydrophobicity', ...
        'normalized_vander_waals', 'polarity', 'polarizability', ...
        'secondary_structure', 'solvent_accessibility'};
    group_name = {'group1', 'group2', 'group3'};
    residue_name = {'residue0%', 'residue25%', 'residue50%', ...
        'residue75%', 'residue100%'};
    len_seq = length(seq);
    d_vector = [];
    ft_names = [];
    for i = 1:7
        property = property_group{i};
        for j = 1:3
            group = property{j};
            index_aa = find(ismember(seq, group));
            len_common = length(index_aa);
            if len_common == 0
                d_vector = [d_vector, zeros(1,5)];
            else
                residue0 = 100*index_aa(1)/len_seq;
                if floor(len_common*0.25) == 0 
                    residue25 = residue0;
                else
                    residue25 = 100*index_aa(floor(len_common*0.25))/len_seq;
                end
                if floor(len_common*0.5) == 0 
                    residue50 = residue0;
                else
                    residue50 = 100*index_aa(floor(len_common*0.5))/len_seq;
                end
                if floor(len_common*0.75) == 0 
                    residue75 = residue0;
                else
                    residue75 = 100*index_aa(floor(len_common*0.75))/len_seq;
                end
                residue100 = 100*index_aa(floor(len_common))/len_seq;
                d_vector = [d_vector, residue0, residue25, ...
                    residue50, residue75, residue100];
            end
            ft_names = [ft_names, ...
                strcat(property_name{i},'_',group_name{j},'_',residue_name)];
        end
    end
     
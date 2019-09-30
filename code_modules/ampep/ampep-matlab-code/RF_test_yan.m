% generate the prediction and score based on input training and testing
% data by random forest with 100 trees.
function [pre,sco]=RF_test_yan(training_data, testing_data)
    test=testing_data; 
    train=training_data;
    tb=TreeBagger(100,train(:,1:end-1),train(:,end));
    [prediction,score,cost]=predict(tb,test);
    pre=str2num(cell2mat(prediction));
    sco=score(:,2);
    
    
%     % calculate area under receiver operating characterisitic curve
%     [X,Y,T,AUC_ROC]=perfcurve(ori,sco(:,2),1);
%     % calculate area under receiver operating characterisitic curve
%     [Xpr,Ypr,Tpr,AUC_pr] = perfcurve(ori, sco(:,2), 1, 'xCrit', 'reca', 'yCrit', 'prec');
%     % calculate tp(true positive), tn(true negative), ...
%     %   fp(false positive), fn(false negative)
%     [tp,usel]=size(find(pre(find(ori(:,1)==1),1)==1));
%     [tn,usele]=size(find(pre(find(ori(:,1)==0),1)==0));
%     [fp,useles]=size(find(pre(find(ori(:,1)==0),1)==1));
%     [fn,useless]=size(find(pre(find(ori(:,1)==1),1)==0));
%     % Sensitivity
%     Sn=tp/(tp+fn);
%     % Specificity
%     Sp=tn/(fp+tn);
%     % Accuarcy
%     Acc=(tp+tn)/(tp+tn+fp+fn);
%     % Matthews correlation coefficient
%     Mcc=((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(fp+tn)*(tn+fn));
%     po=Acc;
%     pe=[(tp+fn)*(tp+fp)+(tn+fn)*(tn+fp)]/(tp+fn+fp+tn)^2;
%     % Cohen's Kappa coefficient
%     kappa=1-(1-po)*(1-pe);
%     recall=Sn;
%     precision=tp/(tp+fp);
%     % return all the measurements
%     Rs_test=[Sn,Sp,Acc,Mcc,AUC_ROC,AUC_pr,kappa];
%     % if you want change the result to confusion matrix and recall,
%     % precision use Rs1_test as result
%     Rs1_test=[tn,tp,fp,fn,recall,precision];
%     % confusion matrix
%     confmat=[tp fn; fp tn];
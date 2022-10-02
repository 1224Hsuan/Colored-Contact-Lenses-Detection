cclds = imageDatastore('CCLdata\train','IncludeSubfolders',true,'LabelSource','foldernames');
cclActual=cclds.Labels;
cclds = augmentedImageDatastore([227 227],cclds);


deepnet = alexnet;
ly=deepnet.Layers;
ly(end-2) = fullyConnectedLayer(2);
ly(end) = classificationLayer;

opts = trainingOptions('sgdm','InitialLearnRate', ...
        0.001,'MaxEpochs',19,'VerboseFrequency',1, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'ExecutionEnvironment','gpu');

[cclnet,info] = trainNetwork(cclds, ly, opts);
%plot( info.TrainingLoss );
%hold on;
%plot(info.TrainingAccuracy);
%legend('TrainingLoss','TrainingAccuracy');

testds = imageDatastore('CCLdata\validation','IncludeSubfolders',true,'LabelSource','foldernames');
testActual=testds.Labels;
testds = augmentedImageDatastore([227 227],testds);

testpreds = classify(cclnet,testds);



cclpreds = classify(cclnet,testds);
numCorrect=nnz(cclpreds==testActual);
fracCorrect=numCorrect/numel(cclpreds);
fprintf('%.4f\n',fracCorrect);
fid=importdata('CCL_predict.csv');


test_ds = imageDatastore('CCLdata\test');
test_ds = augmentedImageDatastore([227 227],test_ds);
testpreds = classify(cclnet,test_ds);

fid_out=fopen('CCL_test5.csv','w');
[r,c]=size(fid);
	fprintf(fid_out,fid{1});

 	fprintf(fid_out,'\n');
 for k=2:r
  fprintf(fid_out,fid{k});
	if (testpreds(k-1)=='With_CCL')
 		fprintf(fid_out,'1');
	else
 		fprintf(fid_out,'0');
	end
  fprintf(fid_out,'\n');
 end
fclose(fid_out);
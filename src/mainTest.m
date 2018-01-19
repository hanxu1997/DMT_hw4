load('parms/Wc.mat', 'Wc');
load('parms/bc.mat', 'bc');
load('parms/Wd.mat', 'Wd');
load('parms/bd.mat', 'bd');

% load('..\output\S4\testImages.mat','testImages');
% %test data ≤‚ ‘ºØ
% addpath ./common/;
% testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
% testImages = reshape(testImages,imageDim,imageDim,[]);
% testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
% testLabels(testLabels==0) = 10; % Remap 0 to 10

imageDim = 28; % ÕºœÒ≥ﬂ¥Á
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension (should divide imageDim-filterDim+1)

activations = cnnConvolve(filterDim, numFilters, testImages, Wc, bc);%sigmoid(wx+b)
activationsPooled = cnnPool(poolDim, activations);
activationsPooled = reshape(activationsPooled,[],17);
h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1));
[~,preds] = max(probs,[],1);
preds = preds';
% activations = cnnConvolve(filterDim, numFilters, testImages, Wc, bc);%sigmoid(wx+b)
% activationsPooled = cnnPool(poolDim, activations);
% activationsPooled = reshape(activationsPooled,[],length(testLabels));
% h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
% probs = bsxfun(@rdivide,h,sum(h,1));
% [~,preds] = max(probs,[],1);
% preds = preds';
% acc = sum(preds==testLabels)/length(preds);
% fprintf('Accuracy is %f\n',acc);
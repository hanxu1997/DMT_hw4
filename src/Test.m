close all;
clc;
clear;
num1 = rgb2gray(imread('..\test\2.jpg'));
num2 = rgb2gray(imread('..\test\4.jpg'));
num1 = imresize(num1,[28 28]);
num2 = imresize(num2,[28 28]);
testImages = zeros(28,28,2);
testImages(:,:,1) = num1;
testImages(:,:,2) = num2;
load('parms/Wc.mat', 'Wc');
load('parms/bc.mat', 'bc');
load('parms/Wd.mat', 'Wd');
load('parms/bd.mat', 'bd');
imageDim = 28; % Í¼Ïñ³ß´ç
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension (should divide imageDim-filterDim+1)

activations = cnnConvolve(filterDim, numFilters, testImages, Wc, bc);%sigmoid(wx+b)
activationsPooled = cnnPool(poolDim, activations);
activationsPooled = reshape(activationsPooled,[],2);
h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1));
[~,preds] = max(probs,[],1);
preds = preds';

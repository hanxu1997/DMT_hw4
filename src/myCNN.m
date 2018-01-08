%% STEP 0: Initialize Parameters and Load Data

imageDim = 28; % 图像尺寸
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension (should divide imageDim-filterDim+1)

% Load MNIST Train 训练集
%training data
addpath ./common/;
images = loadMNISTImages('./common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('./common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%test data 测试集
testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%Initialize Parameters
% 权重初始化为均值0方差0.1的高斯分布
% 偏置值初始化为0
Wc = 1e-1*randn(filterDim,filterDim,numFilters);
bc = zeros(numFilters, 1);
% 卷积层的输出图像的尺寸
outDim = imageDim - filterDim + 1; % dimension of convolved image
outDim = outDim/poolDim; % 池化层输出图像的尺寸
hiddenSize = outDim^2*numFilters; % 隐藏层的尺寸
% 初始化全连接层权重与偏置
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
bd = zeros(numClasses, 1);

%% STEP 1: Learn Parameters+
epochs = 3; % 分批处理
alpha = 1e-1; % 学习速率0.1
minibatch = 256; % 每批样本个数
% Setup for momentum
mom = 0.5; % 动量
momentum = .95;
momIncrease = 20;
% 速度
Wc_velocity = zeros(size(Wc));
Wd_velocity = zeros(size(Wd));
bc_velocity = zeros(size(bc));
bd_velocity = zeros(size(bd));
lambda = 0.0001; % weight decay factor
m = length(labels);

%% SGD loop 随机梯度下降（小批量梯度下降）
% 因为是采用的mini-batch梯度下降法，每次循环中权值已经迭代多次
% 所以总共对样本的循环次数次数比标准梯度下降法要少很多
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    % 返回一个1维数组（向量），该数组是整数1到60000的一组随机排列，彼此无重复，顺序随机
    rp = randperm(m);
    % 步长256
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = momentum;
        end;

        % get next randomly selected minibatch
        mb_data = images(:,:,rp(s:s+minibatch-1)); %28*28*256
        mb_labels = labels(rp(s:s+minibatch-1)); % 256*1

        % evaluate the objective function on the next minibatch
        numImages = length(mb_labels); % 256
        convDim = imageDim-filterDim+1; % dimension of convolved output:20
        outputDim = (convDim)/poolDim; % dimension of subsampled output:10
        %% 反向传播的训练过程来调整过滤器值（权重）
        % 1.Feedforward Propagation 前向传导
        activations = cnnConvolve(filterDim, numFilters, mb_data, Wc, bc);%sigmoid(wx+b)
        activationsPooled = cnnPool(poolDim, activations);
        activationsPooled = reshape(activationsPooled,[],numImages);
        h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
        % probs：对所有图像所属分类的预测结果，每一列对应一张图像，一共10行，第i行代表这张图像属于第i类的概率
        probs = bsxfun(@rdivide,h,sum(h,1));
        
        % 2.Caculate Cost 损失函数
        logp = log(probs);
        index = sub2ind(size(logp),mb_labels',1:size(probs,2));
        ceCost = -sum(logp(index));
        wCost = lambda/2 * (sum(Wd(:).^2)+sum(Wc(:).^2));
        cost = ceCost/numImages + wCost;
        
        % 3.Backpropagation 后向传导
        output = zeros(size(probs));
        output(index) = 1;
        DeltaSoftmax = probs - output; % 输出层的灵敏度

        DeltaPool = reshape(Wd' * DeltaSoftmax,outputDim,outputDim,numFilters,numImages);
        DeltaUnpool = zeros(convDim,convDim,numFilters,numImages);
        % 求池化层的灵敏度，用kron函数实现上采样
        for imNum = 1:numImages
            for FilterNum = 1:numFilters
                unpool = DeltaPool(:,:,FilterNum,imNum);
                DeltaUnpool(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim))./(poolDim ^ 2);
            end
        end
        % 卷积层的灵敏度
        DeltaConv = DeltaUnpool .* activations .* (1 - activations);
        % 计算梯度
        Wd_grad = (1./numImages) .* DeltaSoftmax*activationsPooled'+lambda*Wd;
        bd_grad = (1./numImages) .* sum(DeltaSoftmax,2);

        bc_grad = zeros(size(bc));
        Wc_grad = zeros(filterDim,filterDim,numFilters);

        for filterNum = 1:numFilters
            error = DeltaConv(:,:,filterNum,:);
            bc_grad(filterNum) = (1./numImages) .* sum(error(:));
        end
        % 卷积层的灵敏度
        for filterNum = 1:numFilters
            for imNum = 1:numImages
                error = DeltaConv(:,:,filterNum,imNum);
                DeltaConv(:,:,filterNum,imNum) = rot90(error,2);
            end
        end

        for filterNum = 1:numFilters
            for imNum = 1:numImages
                Wc_grad(:,:,filterNum) = Wc_grad(:,:,filterNum) + conv2(mb_data(:,:,imNum),DeltaConv(:,:,filterNum,imNum),'valid');
            end
        end
        Wc_grad = (1./numImages) .* Wc_grad + lambda*Wc;
        % 4.参数更新
        %%% YOUR CODE HERE %%%
        Wc_velocity = mom*Wc_velocity+alpha*Wc_grad;
        Wc = Wc - Wc_velocity;
        Wd_velocity = mom*Wd_velocity+alpha*Wd_grad;
        Wd = Wd - Wd_velocity;
        bc_velocity = mom*bc_velocity+alpha*bc_grad;
        bc = bc - bc_velocity;
        bd_velocity = mom*bd_velocity+alpha*bd_grad;
        bd = bd - bd_velocity;
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;

    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;
    
    %% test at each epoch
    activations = cnnConvolve(filterDim, numFilters, testImages, Wc, bc);%sigmoid(wx+b)
    activationsPooled = cnnPool(poolDim, activations);
    activationsPooled = reshape(activationsPooled,[],length(testLabels));
    h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
    probs = bsxfun(@rdivide,h,sum(h,1));
    [~,preds] = max(probs,[],1);
    preds = preds';
    acc = sum(preds==testLabels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
end;
% save('parms/Wc.mat', 'Wc');
% save('parms/bc.mat', 'bc');
% save('parms/Wd.mat', 'Wd');
% save('parms/bd.mat', 'bd');



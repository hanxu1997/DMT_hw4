clear
close all
clc
folder_image = '..\output\S2\cut';
filelist = U1_ReadFileList('..\output\S3\cut.txt');
filenum = length(filelist);

% 待存入文件夹
folder_bw = '..\output\S4\bw';
folder_pad = '..\output\S4\padding';
folder_nums = '..\output\S4\nums';
% 如果没有则创建
if ~exist(folder_bw,'dir')
    mkdir(folder_bw)
end
if ~exist(folder_pad,'dir')
    mkdir(folder_pad)
end
if ~exist(folder_nums,'dir')
    mkdir(folder_nums)
end
testImages = zeros(28,28,17);
for i = 1:filenum
    filename = filelist{i}(1:end-4);
    grayimg = rgb2gray(imread(fullfile(folder_image,filelist{i})));
    % ---------------------------------------------------------------------
    bwImg = im2bw(grayimg, 0.65);
    bwImg = F2_imerode(bwImg);
    bwfilename = sprintf('%s_bw.jpg',filename);
    file_bw = fullfile(folder_bw,bwfilename);
    imwrite(bwImg, file_bw);
    % ---------------------------------------------------------------------
    bwImg = ~bwImg;
    % 矩阵扩展为方形
    [m,n] = size(bwImg);
    dim = max([m,n]);
    padding = zeros(dim);
    x = floor((dim-m)/2)+1;
    y = floor((dim-n)/2)+1;
    padding(x:x+m-1,y:y+n-1) = bwImg;
    padding = bwareaopen(padding,100,8);
    padfilename = sprintf('%s_pad.jpg',filename);
    file_pad = fullfile(folder_pad,padfilename);
    imwrite(padding, file_pad);
    
    resizeImg = imresize(padding,[24 24]);
    C = zeros(28);
    C(3:26,3:26) = resizeImg;
    file_nums = fullfile(folder_nums,filelist{i});
%     resizeImg = ~resizeImg;
    imwrite(C, file_nums);
    testImages(:,:,i) = C;
end
save('..\output\S4\testImages.mat','testImages');

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
    % 旋转
    switch i
        case 1
            padding = imrotate(padding,-20,'bicubic','crop');
        case 5
            padding = imrotate(padding,-4,'bicubic','crop');
        case 10
            padding = imrotate(padding, 2,'bicubic','crop');
        case 15
            padding = imrotate(padding, -10,'bicubic','crop');
        otherwise
            
    end
    padfilename = sprintf('%s_pad.jpg',filename);
    file_pad = fullfile(folder_pad,padfilename);
    imwrite(padding, file_pad);
    switch i
        case 5
            resizeImg = imresize(padding,[22 28]);
            C = zeros(28);
            C(6:27,1:28) = resizeImg;
        case 10
            resizeImg = imresize(padding,[22 24]);
            C = zeros(28);
            C(5:26,4:27) = resizeImg;  
        case 15
            resizeImg = imresize(padding,[22 24]);
            C = zeros(28);
            C(5:26,3:26) = resizeImg;
        case 16
            resizeImg = imresize(padding,[22 24]);
            C = zeros(28);
            C(5:26,4:27) = resizeImg;   
        otherwise
            resizeImg = imresize(padding,[22 22]);
            C = zeros(28);
            C(6:27,6:27) = resizeImg;
    end

    file_nums = fullfile(folder_nums,filelist{i});
%     resizeImg = ~resizeImg;
    imwrite(C, file_nums);
    testImages(:,:,i) = C;
end
save('..\output\S4\testImages.mat','testImages');

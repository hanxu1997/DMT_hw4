clear
close all
clc

% 测试数据集filelist生成
filelist_name = 'cut.txt';
folder_source = '..\output\S2\cut';
% filelist存储文件夹
folder_filelist = '..\output\S3';

% 不同类型图片
% 打开待写入txt文件
fid = fopen(fullfile(folder_filelist,filelist_name),'w+');
    filelist = dir(fullfile(folder_source,['*' '.jpg']));
    filenum = length(filelist);
    filelist = F23_SortFileListByNumber(filelist,'.jpg');
    for j=1:filenum
        % 逐行写入[序号 文件名]
        fprintf(fid,'%d %s\n',j,filelist(j).name);
    end

fclose(fid);

clear
close all
clc

% �������ݼ�filelist����
filelist_name = 'cut.txt';
folder_source = '..\output\S2\cut';
% filelist�洢�ļ���
folder_filelist = '..\output\S3';

% ��ͬ����ͼƬ
% �򿪴�д��txt�ļ�
fid = fopen(fullfile(folder_filelist,filelist_name),'w+');
    filelist = dir(fullfile(folder_source,['*' '.jpg']));
    filenum = length(filelist);
    filelist = F23_SortFileListByNumber(filelist,'.jpg');
    for j=1:filenum
        % ����д��[��� �ļ���]
        fprintf(fid,'%d %s\n',j,filelist(j).name);
    end

fclose(fid);

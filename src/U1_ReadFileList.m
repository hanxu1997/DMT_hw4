% Filelist Train.txt
% 1 BSD_100075.jpg
% 2 BSD_100080.jpg
% ……
% 320 ……
function filelist = U1_ReadFileList( fn_list )
    fid = fopen(fn_list,'r');
    C = textscan(fid,'%d %s\n');
    fclose(fid);
    % 取文件名列表
    filelist = C{2};
end


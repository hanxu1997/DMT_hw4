close all;
clc;
clear;
img  = imread('..\input\input.jpg');

% 1. ��ͼ��ת�ɻҶ�ͼ
img_gray = rgb2gray(img);
[m,n] = size(img_gray);
figure(1);
subplot(1,2,1);
imshow(img_gray);
imwrite(img_gray,'..\output\S1\img_gray.jpg');
title('�Ҷ�ͼ');
% 2. ��Ե���
edged_result = edge(img_gray,'sobel');
subplot(1,2,2);
% ɾ��С�������
edged_result = bwareaopen(edged_result,100,8);
% ����
edged_result = F1_imdilate(edged_result);
imshow(edged_result);
title('��Ե���');
imwrite(edged_result,'..\output\S1\edge.jpg');

% 3. ����任�õ�A4ֽ��Ե
% �����ֵͼ��ı�׼����任
[H,Theta,Rho] = hough(edged_result);
% �ӻ���任����H����ȡ7����ֵ��
P  = houghpeaks(H,7,'threshold',0.4*max(H(:)));
x = Theta(P(:,2)); 
y = Rho(P(:,1));
% ��ԭͼ�е���
lines = houghlines(edged_result,Theta,Rho,P,'FillGap',1000,'MinLength',40);
figure(2);
subplot(1,2,1);
imshow(img);
title('�ɱ�Ե�ҳ��ĸ��ǵ�');
hold on
for k = 1:length(lines)
 xy = [lines(k).point1; lines(k).point2];
 plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
 hold on
end


% 4. ͨ��A4ֽ�ı߼���A4ֽ���ĸ��ǵ�
line_14 = [ lines(1).point1;lines(1).point2;lines(4).point1;lines(4).point2];
A = F0_intersection(line_14);
line_13 = [ lines(1).point1;lines(1).point2;lines(3).point1;lines(3).point2];
B = F0_intersection(line_13);
line_32 = [ lines(3).point1;lines(3).point2;lines(2).point1;lines(2).point2];
C = F0_intersection(line_32);
line_24 = [ lines(2).point1;lines(2).point2;lines(4).point1;lines(4).point2];
D = F0_intersection(line_24);
A = round(A);
B = round(B);
C = round(C);
D = round(D);
plot(A(1),A(2),'.','markersize',20,'color','b');
plot(B(1),B(2),'.','markersize',20,'color','b');
plot(C(1),C(2),'.','markersize',20,'color','b');
plot(D(1),D(2),'.','markersize',20,'color','b');

% ͸�ӱ任��ͼ��
original = [A;B;C;D];
new = [1 1;1 m;n m;n 1];
TForm = cp2tform(original,new,'projective');
figure(2);
subplot(1,2,2);
output_img = imtransform(img,TForm,'XData',[1 n], 'YData',[1 m]);
imshow(output_img);
title('͸�ӱ任��ͼ��');
imwrite(output_img,'..\output\S1\S1_result.jpg');






function output_img = F1_imdilate( input_img )
    B = [ 0 1 0
          1 1 1
          0 1 0];
    % ��һ�θ�ʴ
    output_img = imdilate(input_img,B);
    % �ڶ��θ�ʴ
    output_img = imdilate(output_img,B);
end


function output_img = F2_imerode( input_img )
    B = [ 0 1 0
          1 1 1
          0 1 0];
    % ��һ�θ�ʴ
    output_img = imerode(input_img,B);
    % �ڶ��θ�ʴ
    output_img = imerode(output_img,B);
end


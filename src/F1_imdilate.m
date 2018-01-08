function output_img = F1_imdilate( input_img )
    B = [ 0 1 0
          1 1 1
          0 1 0];
    % 第一次腐蚀
    output_img = imdilate(input_img,B);
    % 第二次腐蚀
    output_img = imdilate(output_img,B);
end


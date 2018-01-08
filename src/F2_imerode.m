function output_img = F2_imerode( input_img )
    B = [ 0 1 0
          1 1 1
          0 1 0];
    % 第一次腐蚀
    output_img = imerode(input_img,B);
    % 第二次腐蚀
    output_img = imerode(output_img,B);
end


Bands = [15 17 19 21 24]
Bandstr = ["15" "17" "19" "21" "24"]

%%%%%%%%%%%%%%%%%%% Fruits

Main_dir_results = './Data/Fruit/test_results/'
Main_dir_inputs = './Data/Fruit/test_inputs/'
Main_dir_labels = './Data/Fruit/test_labels/'
images_list = ["CompData (19)_LED_" "CompData (117)_FLU_"]


for image = 1:1:2
    fig = figure
    load(Main_dir_inputs+images_list(image), 'CompData');
    input = CompData;
    load(Main_dir_labels+images_list(image), 'CompData');
    label = CompData;
    load(Main_dir_results+images_list(image), 'rad');
    result = rad;
    for i=1:1:5
    subplot(3,5,i); imshow(input(:,:,Bands(i)))
    title('Input Band'+Bandstr(i))
    subplot(3,5,i+5); imshow(result(:,:,Bands(i))*6)
    title('Result Band'+Bandstr(i))
    subplot(3,5,i+10); imshow(label(:,:,Bands(i)))
    title('GT Band'+Bandstr(i))
    end
    f = gcf
    exportgraphics(f,images_list(image) +'Comparison'+'.png','Resolution',1500)
%     save(fig,images_list(image) +'Comparison','.jpg');
end

%%%%%%%%%%%%%%%%%%%%%%%Material

Main_dir_results = './Data/Material/test_results/'
Main_dir_inputs = './Data/Material/test_inputs/'
Main_dir_labels = './Data/Material/test_labels/'
images_list = ["A_Input (1)" "B_Input_LED (24)"]


for image = 1:1:2
    f = figure
    load(Main_dir_inputs+images_list(image), 'CompData');
    input = CompData;
    load(Main_dir_labels+images_list(image), 'CompData');
    label = CompData;
    load(Main_dir_results+images_list(image), 'rad');
    result = rad;
    for i=1:1:5
    subplot(3,5,i); imshow(input(:,:,Bands(i)))
    title('Input Band'+Bandstr(i))
    subplot(3,5,i+5); imshow(result(:,:,Bands(i))*6)
    title('Result Band'+Bandstr(i))
    subplot(3,5,i+10); imshow(label(:,:,Bands(i)))
    title('GT Band'+Bandstr(i))
    end
    exportgraphics(f,images_list(image) +'Comparison'+'.png','Resolution',1500)
%     save(fig,images_list(image) +'Comparison','.jpg');
end

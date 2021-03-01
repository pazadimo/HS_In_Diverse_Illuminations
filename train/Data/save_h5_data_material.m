global FILE_COUNT;
global TOTALCT;
global CREATED_FLAG;

string='train';
% string='valid';
if strcmp(string, 'train') == 1

    LABEL_dir = '\Data\Material\Label_train\';
    labels=dir(fullfile(LABEL_dir,'*.mat'));

    INPUT_dir = '\Data\Material\Input_train\';
    inputs=dir(fullfile(INPUT_dir,'*.mat'));
    
    order= randperm(size(labels,1));
   
else

    LABEL_dir = '.\Material\Label_valid\';
    labels=dir(fullfile(LABEL_dir,'*.mat'));

    INPUT_dir = '.\Material\Input_valid\';
    inputs=dir(fullfile(INPUT_dir,'*.mat'));
    
    order= randperm(size(labels,1));
end  

%% Initialization the patch and stride
size_input=50;
size_label=50;
label_dimension=25;
data_dimension=25;
stride=50;


%% Initialization the hdf5 parameters
% prefix=[string '_Valid_ALL_'];
prefix=[string '_Material2_'];
chunksz=64;
TOTALCT=0;
FILE_COUNT=0;
amount_hd5_image=100000;
CREATED_FLAG=false;
c=0;

%% For loop  RGB-HS-HD5  
for i=1:size(labels,1)
     if mod(i,amount_hd5_image)==1     
         filename=getFilename(labels(order(i)).name,prefix,LABEL_dir);
     end

    name_label=strcat(LABEL_dir,labels(order(i)).name); 
    name_input=strcat(INPUT_dir,inputs(order(i)).name); 
    
    label_temp=struct2cell(load(name_label,'label'));
    if(isempty(label_temp)) 
        label_temp=struct2cell(load(name_label,'CompData'));
    end
    hs_label=cell2mat(label_temp);
    
%     hs_label=hs_label/(2^12-1);
    hs_label=hs_label/(20.0);

    input_temp=struct2cell(load(name_input,'input'));
    if(isempty(input_temp)) 
        input_temp=struct2cell(load(name_input,'CompData'));
        c = c+1
    end
    hs_input=cell2mat(input_temp);
%     hs_input=hs_input/(2^12-1);
    hs_input=hs_input/(20.0);
    i
    disp(name_input)
    disp(name_label)
%     input_name=[ INPUT_dir 'REFLECTANCE_' labels(order(i)).name(1:end-4) '.png'];
%     
%     input_data_uint=imread(input_name);
%     input_data=im2double(input_data_uint);
       
%     for j=1:label_dimension
%            ConvertHStoNbands(input_data,hs_label(:,:,j),size_input,size_label,1,data_dimension,stride,chunksz,amount_hd5_image,filename)
%     end
    ConvertHStoNbands(hs_input,hs_label,size_input,size_label,label_dimension,data_dimension,stride,chunksz,amount_hd5_image,filename)

end       
 

function filename_change=getFilename(filename,prefix,folder_label)
       filename_change=prefix
%        [prefix 'Fruit'];
%        filename_change=filename;
%        filename_change=[folder_label prefix filename_change];
%        filename_change=filename_change(1:end-4);
       filename_change=strcat(filename_change,'.h5');
end
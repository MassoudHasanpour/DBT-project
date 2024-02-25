topLevelFolder = pwd;
files = dir(topLevelFolder);
dirFlags = [files.isdir];
subFolders = files(dirFlags);
subFolders = subFolders(~ismember({subFolders(:).name},{'.','..'}));
Patients = {subFolders(:).name};

j = 1;
Hist_Ab_Br = [];
Patients_done = {};

for i = 1:length(Patients)
    SubF = append('D:\Projects\DBT project\Patients with benign cancer' ...
        , '/', Patients{i});
    list = dir(SubF);
    list = list(~ismember({list(:).name},{'.','..'}));
    if length(list) == 1 
        list = dir(list(1).folder);
        list = list(~ismember({list(:).name},{'.','..'}));
        SubF = append(list(1).folder, '\' , list(1).name);
        
        list = dir(SubF);
        list = list(~ismember({list(:).name},{'.','..'}));
        try
            if length(list) == 1 || length(list) == 2
                SubF = append(list(1).folder, '\' , list(1).name);

                SubF_im = append(SubF, '/', '1-1_crrct.dcm');
                SubF_abmask = append(SubF, '/', '1-1_mask.dcm');
                SubF_brmask = append(SubF, '/', '1-1_Breast_mask.dcm');

                Im = dicomread(SubF_im);
                Ab_mask = dicomread(SubF_abmask);
                Br_mask = dicomread(SubF_brmask);

                Im_Ab = Im(Ab_mask(:)>0);
        %         Im_Ab = Im_Ab(Im_Ab > 0 );
                
                histogram(Im_Ab,"NumBins",100,'Normalization','probability', "BinEdges", [240:4:640])
                axisHandle = gca;     
                histHandle = axisHandle.Children;   
                barHeight = histHandle.Values; 
                Hist_Ab_Br(j,300) = 0;
                Patients_done{j} = Patients{i};
                j = j+1;
                Hist_Ab_Br(j,1:100) = barHeight;
                
                Im_Br = Im(Br_mask(:) > 0);
        %         Im_Br = Im_Ab(Im_Ab > 0 );

                histogram(Im_Br, "NumBins",200,'Normalization','probability', "BinEdges", [240:2:640])
                axisHandle = gca;     
                histHandle = axisHandle.Children;   
                barHeight = histHandle.Values; 
                Hist_Ab_Br(j, 101:end) = barHeight;
                
            elseif length(list) > 2
                
                SubF = append(list(1).folder, '\' , list(1).name);

                SubF_im = append(SubF, '/', '1-1_crrct.dcm');
                SubF_abmask = append(SubF, '/', '1-1_mask.dcm');
                SubF_brmask = append(SubF, '/', '1-1_Breast_mask.dcm');

                Im = dicomread(SubF_im);
                Ab_mask = dicomread(SubF_abmask);
                Br_mask = dicomread(SubF_brmask);

                Im_Ab = Im(Ab_mask(:)>0);
        %         Im_Ab = Im_Ab(Im_Ab > 0 );
                
                
                histogram(Im_Ab,"NumBins",100,'Normalization','probability', "BinEdges", [240:4:640])
                axisHandle = gca;     
                histHandle = axisHandle.Children;   
                barHeight = histHandle.Values; 
                Hist_Ab_Br(j,300) = 0;
                Patients_done{j} = Patients{i};
                j = j+1;
                Hist_Ab_Br(j,1:100) = barHeight;
                
                
                Im_Br = Im(Br_mask(:) > 0);
        %         Im_Br = Im_Ab(Im_Ab > 0 );

                histogram(Im_Br, "NumBins",200,'Normalization','probability', "BinEdges", [240:2:640])
                axisHandle = gca;     
                histHandle = axisHandle.Children;   
                barHeight = histHandle.Values; 
                Hist_Ab_Br(j, 101:end) = barHeight;
                
                %%%%%%%%%%%%%%%%%%%
                SubF = append(list(3).folder, '\' , list(3).name);
                Patients_done{j} = Patients{i};
                SubF_im = append(SubF, '/', '1-1_crrct.dcm');
                SubF_abmask = append(SubF, '/', '1-1_mask.dcm');
                SubF_brmask = append(SubF, '/', '1-1_Breast_mask.dcm');

                Im = dicomread(SubF_im);
                Ab_mask = dicomread(SubF_abmask);
                Br_mask = dicomread(SubF_brmask);

                
                Im_Ab = Im(Ab_mask(:)>0);
        %         Im_Ab = Im_Ab(Im_Ab > 0 );
                
                histogram(Im_Ab,"NumBins",100,'Normalization','probability', "BinEdges", [240:4:640])
                axisHandle = gca;     
                histHandle = axisHandle.Children;   
                barHeight = histHandle.Values; 
                Hist_Ab_Br(j,300) = 0;
                Patients_done{j} = Patients{i};
                j = j+1;
                Hist_Ab_Br(j,1:100) = barHeight;   
                
                
                Im_Br = Im(Br_mask(:) > 0);
        %         Im_Br = Im_Ab(Im_Ab > 0 );

                histogram(Im_Br, "NumBins",200,'Normalization','probability', "BinEdges", [240:2:640])
                axisHandle = gca;     
                histHandle = axisHandle.Children;   
                barHeight = histHandle.Values; 
                Hist_Ab_Br(j, 101:end) = barHeight;
            end
        catch 
            disp(list(1).folder)
            
        end
        
    elseif length(list) > 1
        disp(list(1).folder)
    end
    disp(i)

end
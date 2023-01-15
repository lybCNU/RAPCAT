%分析atlas中各cell的位置变化统计信息
%   1.以每一条train100_valid线虫为模板，将其他线虫仿射对齐到模板线虫，取平均获得100个初始atlas,分析cell位置变化std，
%   2.使用分段仿射对齐将train100_valid映射到每一个atlas中，计算fidelity，取fidelity平均值最高的为最终atlas
%   3.保存新的atlas
clc
clear all
close all
addpath('./matlab_io_basicdatatype/');

 filepath='./traindata/train100/';
 L=dir([filepath,'/*.apo']);
 for k=1:length(L)
     %读入训练数据
     filename=[filepath,'/',L(k).name];
     apo_data=load_v3d_pointcloud_file(filename);
     arr_apo_data{k}=apo_data;
     clear apo_data
     fprintf('[%d]:%s\n',k,L(k).name);
 end
 
 for iii=1:length(arr_apo_data)
     for jjj=1:length(arr_apo_data)
        apo_tar=arr_apo_data{iii};    apo_sub=arr_apo_data{jjj};
        X_tar=zeros(3,size(apo_tar,2));     X_sub=X_tar;
        for i=1:size(apo_tar,2)
            X_tar(1,i)=apo_tar{i}.x;
            X_tar(2,i)=apo_tar{i}.y;
            X_tar(3,i)=apo_tar{i}.z;
            X_sub(1,i)=apo_sub{i}.x;
            X_sub(2,i)=apo_sub{i}.y;
            X_sub(3,i)=apo_sub{i}.z;
        end
        T=affine3D_model(X_sub,X_tar);  %T*X_sub=X_tar
        X_sub2tar=T*[X_sub(1:3,:);ones(1,size(X_sub,2))];
        arr_man2atlas_aff{iii}{jjj}=X_sub2tar(1:3,:);
        clear T X_sub2tar X_tar X_sub apo_tar apo_sub;
     end
fprintf('Affine to [%d]:%s\n',iii,L(iii).name);
 end

 ncell=558;
 for iii=1:length(arr_man2atlas_aff)
     for jjj=1:ncell
        arr_1cellpos_aff=[];
        for kkk=1:length(arr_man2atlas_aff)
            arr_1cellpos_aff(:,kkk)=arr_man2atlas_aff{iii}{kkk}(:,jjj);
        end
     pos_mean_aff=mean(arr_1cellpos_aff,2);
     arr_pos_mean_aff{iii}(:,jjj)=pos_mean_aff;
     pos_std_aff=std(arr_1cellpos_aff,1,2);
     arr_pos_std_aff{iii}(:,jjj)=pos_std_aff;
     clear pos_std_aff pos_mean_aff arr_1cellpos_aff;
     end
fprintf('cal mean and std [%d]:%s\n',iii,L(iii).name);
 end

for iii=1:length(arr_pos_mean_aff)
    for jjj=1:length(arr_apo_data)
        apo_tar=arr_pos_mean_aff{iii};
        apo_tar_std=arr_pos_std_aff{iii};
        apo_sub=arr_apo_data{jjj};
        X_tar=apo_tar;
        X_sub=X_tar;
        for i=1:size(apo_tar,2)
            X_sub(1,i)=apo_sub{i}.x;
            X_sub(2,i)=apo_sub{i}.y;
            X_sub(3,i)=apo_sub{i}.z;
        end
        %piecewise affine align sub to atlas
        xmin=min(X_sub(1,:));
        xmax=max(X_sub(1,:));
        piecesize=(xmax-xmin)/8;
        piecestep=piecesize/8;
        npiece=0;
        for step=0:100
            npiece=npiece+1;
            %find all point within current segment/piece
            xmin_piece=xmin+step*piecestep;
            xmax_piece=xmin_piece+piecesize;
            ind=find(X_sub(1,:)>=xmin_piece & X_sub(1,:)<=xmax_piece);
            X_sub_piece=X_sub(:,ind);
            X_tar_piece=X_tar(:,ind);

            %affine align current segment/piece to atlas
            T=affine3D_model(X_sub_piece,X_tar_piece);  %T*X_sub=X_tar
            X_sub2tar_piece=T*[X_sub_piece(1:3,:);ones(1,size(X_sub_piece,2))];

            cellarr_X_sub2tar_piece{npiece}=X_sub2tar_piece(1:3,:);
            cellarr_ind_piece{npiece}=ind;

            if(xmax_piece>xmax) break; end
        end

        % average point in all pieces to obtain the piecewise affine alignment result
        ncell=size(apo_tar,2);
        for i=1:ncell
            cellarr_avg{i}=[];
        end
        for i=1:length(cellarr_ind_piece)
            for j=1:length(cellarr_ind_piece{i})
                ind=cellarr_ind_piece{i}(j);
                cellarr_avg{ind}=[cellarr_avg{ind},cellarr_X_sub2tar_piece{i}(:,j)];
            end
        end
        for i=1:ncell
            X_sub2tar_avg(:,i)=mean(cellarr_avg{i},2);
        end
        xyzdiff_Xsub2tar=X_sub2tar_avg-X_tar;
        exponential_term=0.5*sum(xyzdiff_Xsub2tar.^2./(apo_tar_std.^2));
        shape_prob_pwaff{iii}{jjj}=exp(-exponential_term);
        mean_shape_prob_pwaff(iii,jjj)=mean(exp(-exponential_term));
        clear X_sub X_tar step apo_tar apo_sub xmin xmax piecesize piecestep npiece xmin_piece xmax_piece ind X_sub_piece X_tar_piece T X_sub2tar_piece cellarr_X_sub2tar_piece cellarr_ind_piece cellarr_avg X_sub2tar_avg xyzdiff_Xsub2tar exponential_term;
    end
fprintf('cal shape_prob_pwaff [%d]:%s\n',iii,L(iii).name);
end
clear i iii jjj k kkk j
fprintf('save atlas\n');
filename='./traindata/100-atlas_by_train100_valid1.mat';

atlas_ind=1:100;
for k=1:length(atlas_ind)
    atlas{k}=arr_apo_data{atlas_ind(k)};
    pos_std_aff{k}=arr_pos_std_aff{atlas_ind(k)};
    for i=1:ncell
        atlas{k}{i}.x=arr_pos_mean_aff{atlas_ind(k)}(1,i);
        atlas{k}{i}.y=arr_pos_mean_aff{atlas_ind(k)}(2,i);
        atlas{k}{i}.z=arr_pos_mean_aff{atlas_ind(k)}(3,i);
    end
end
save(filename,'atlas','pos_std_aff');



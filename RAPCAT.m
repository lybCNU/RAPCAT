% C.elegant cell recognition by Lei Qu
%		1. predict cell's ID (cell num = 558)
%       2. given manual recog result, calculate recognition accuracy
%       3. deal cell num < 558
%       4. output cell recog fidelity -> add new term apo.fidelity
%       5. preserve some cell's manual recog result(comment=*FIX*)
% modified by Yongbin Li
%		1. deal cell num slightly more than 558
%		2. use multiple atlas as template for cell recognition
function main(datapath,str_flagID)
	switch nargin
		case 2
			if(strcmpi(str_flagID,'withID')==0)
				str_flagID='noID';
			end
		otherwise
			str_flagID='noID';
	end
    addpath('./RPM/');
    addpath('./matlab_io_basicdatatype/');
    addpath('./bipartite');
    load('./traindata/100-atlas_by_train100_valid.mat') %load all 100 atlas
    selected_atlas_ind=[2 14 29 57 78]; %set selected atlas index
    %%%%%%%'noID'(output recog result no accu), 'withID'(output recog accu)
    %datapath='./input_noID/';   str_flagID='noID';
    %datapath='../testatf6/'; str_flagID='withID';
    % recorded selected_atlas by selected atlas index
    for i=1:length(selected_atlas_ind)
        selected_atlas{i}=atlas{selected_atlas_ind(i)};
        selected_pos_std_aff{i}=pos_std_aff{selected_atlas_ind(i)};
    end
    datapath=[datapath,'/'];outpath=datapath;
    L=dir([datapath,'/*.ano.ano.txt']);
    recogaccu=-1*ones(length(L),1);
    for iter_file=1:length(L)
        filename_sub=L(iter_file).name;
        filepath_apo_sub=[datapath,filename_sub];
        
        fprintf('==========================================================\n');
        fprintf('#(%d):%s\n',iter_file,filename_sub);
        fprintf('==========================================================\n');
        
        fprintf('[1] Load data ...\n');
        % Load sub apo file
        apo_sub=load_v3d_apo_file(filepath_apo_sub);
        apo_sub_bk=apo_sub;
        fprintf('\tload sub apo file:[%s]\n',filename_sub);
        fprintf('\t\t[%d] cells load\n',length(apo_sub));
        clear ind filepath_apo_sub
%         %Load atlas apo file
%         filepath_apo_atlas=atlasapo;
%         apo_atlas=load_v3d_pointcloud_file(filepath_apo_atlas);
          apo_atlas=atlas{1};
%         ind=strfind(filepath_apo_atlas,'/');
%         filename_atlas=filepath_apo_atlas(ind(end)+1:end);
%         fprintf('\tload atlas apo file:[%s]\n',filename_atlas);
%         fprintf('\t\t[%d] cells load\n',length(apo_atlas));
%         clear ind filepath_apo_atlas filename_atlas

        
        %Generate validcell index
        fprintf('[2] Generate validcell index...\n');
        ind_validcell=[];
        for i=1:length(apo_sub)
            cellname=strtrim(apo_sub{i}.name);
            if ~(strcmpi(cellname,'f1') || strcmpi(cellname,'f2') || strcmpi(cellname,'f3') || strcmpi(cellname,'f4') || strcmpi(cellname,'f5') || ...
                    strcmpi(cellname,'f6') || strcmpi(cellname,'f7') || strcmpi(cellname,'f8') || strcmpi(cellname,'f9') || strcmpi(cellname,'f10') || ...
                    ~isempty(strfind(cellname,'nouse')) || ~isempty(strfind(cellname,'NOUSE')))
                   % contains(cellname,'nouse') || contains(cellname,'NOUSE'))
                ind_validcell=[ind_validcell;i];
            end
        end
        if(length(ind_validcell) == 558)
            fprintf('\tAssert success! nvalidcell=558 \n');
        else
            fprintf('\tnvalidcell=%d not equal to 558 --> Need check! but program Go on recog them!\n', length(ind_validcell));
        end
        clear i cellname

        % generate groundtruth atlas2sub matching index according to manual annotation 
        % Note: only for viusalization, debugging and compute accu
        if(strcmpi(str_flagID,'withID'))
            ind_atlas2sub_man=-1*ones(length(ind_validcell),1);
            for m=1:length(apo_atlas)
                cellname_atlas=strtrim(apo_atlas{m}.name);
                bfind=0;
                for n=1:length(apo_sub)
                    cellname=strtrim(apo_sub{n}.name);
                    if(strcmpi(cellname_atlas,cellname))
                        ind_atlas2sub_man(m)=n;
                        bfind=1;break;
                    end
                end
                if(bfind==0)
                    fprintf('\t%s not found in manual apo!\n',cellname_atlas);
                end
            end
            clear m n nvalidcell bfind cellname cellname_atlas
        end

        
        %Find fixed cell index (these cell's annotation need to be fixed during recognition)
        % ind_atlas2sub_fix(:,1)=atlas index,
        % ind_atlas2sub_fix(:,2)=sub index
        fprintf('[3] Generate fixcell index...\n');
        ind_atlas2sub_fix=[];
        for i=1:length(ind_validcell)
            if(strcmpi(apo_sub{ind_validcell(i)}.comment,'*FIX*'))
                cellname=strtrim(apo_sub{ind_validcell(i)}.name);
                if(isempty(cellname))
                    fprintf('ERROR: FIXED CELL NOT PROVIDE CELL NAME!!, return!\n');
                    return;
                end
                %make sure provided cell id in atlas
                for m=1:length(apo_atlas)
                    cellname_atlas=strtrim(apo_atlas{m}.name);
                    if(strcmpi(cellname,cellname_atlas))
                        ind_atlas2sub_fix=[ind_atlas2sub_fix;[m,ind_validcell(i)]];
                        break;
                    end
                end
            end         
        end
        for i=1:size(ind_atlas2sub_fix,1)
            fprintf('\t%dth subCell fixed to %dth atlasCell with ID:%s\n',...
                ind_atlas2sub_fix(i,2),ind_atlas2sub_fix(i,1),strtrim(apo_sub{ind_atlas2sub_fix(i,2)}.name));
        end
    
        fprintf('[4] Do cell recog...\n');
        nvalidcell=length(ind_validcell);
        % reformat valid apo data to arr
        X_sub=-1*ones(3,nvalidcell);
        for i=1:nvalidcell
            X_sub(1,i)=apo_sub{ind_validcell(i)}.x;
            X_sub(2,i)=apo_sub{ind_validcell(i)}.y;
            X_sub(3,i)=apo_sub{ind_validcell(i)}.z;
        end 
        for atlas_ind=1:length(selected_atlas)
            X_tar_oneatlas=-1*ones(3,558);
            apo_atlas_one=selected_atlas{atlas_ind};
            for i=1:558
                X_tar_oneatlas(1,i)=apo_atlas_one{i}.x;
                X_tar_oneatlas(2,i)=apo_atlas_one{i}.y;
                X_tar_oneatlas(3,i)=apo_atlas_one{i}.z;
            end
            X_tar{atlas_ind}=X_tar_oneatlas;
        end
        
        % update ind_atlas2sub_man to valid cell (only for viusalization, debugging and compute accu)
        if(strcmpi(str_flagID,'withID'))
            if(nvalidcell>558)
                ind_atlas2validsub_man=-1*ones(nvalidcell,1);
                for i=1:nvalidcell
                    if(ind_atlas2sub_man(i)<=0) continue; end
                    ind_atlas2validsub_man(i)=find(ind_validcell==ind_atlas2sub_man(i));
                end
            else
                ind_atlas2validsub_man=-1*ones(558,1);
                for i=1:558
                    if(ind_atlas2sub_man(i)<=0) continue; end
                    ind_atlas2validsub_man(i)=find(ind_validcell==ind_atlas2sub_man(i));
                end
            end
%             %plot cell point
%             figure;
%             plot3(X_tar(1,:),X_tar(2,:),X_tar(3,:),'r+','markersize',3); hold on
%             plot3(X_sub(1,:),X_sub(2,:),X_sub(3,:),'bo','markersize',3);
%             for i=1:length(X_tar)
%                 if(ind_atlas2validsub_man(i)<=0)
%                     plot3(X_tar(1,i),X_tar(2,i),X_tar(3,i),'ms','markersize',6,'markerfacecolor','m');
%                 else
%                     plot3([X_tar(1,i);X_sub(1,ind_atlas2validsub_man(i))],...
%                         [X_tar(2,i);X_sub(2,ind_atlas2validsub_man(i))],...
%                         [X_tar(3,i);X_sub(3,ind_atlas2validsub_man(i))]);
%                 end
%             end
%             xlabel('x'),ylabel('y'),zlabel('z')
%             axis('equal'); grid on; set (gca, 'box', 'on');
%             hold off;
        end

        % update ind_atlas2sub_fix to valid cell
        % ind_atlas2validsub_fix(:,1)=atlas index,
        % ind_atlas2validsub_fix(:,2)=valid sub index
        ind_atlas2validsub_fix=ind_atlas2sub_fix;
        for i=1:size(ind_atlas2sub_fix,1)
            ind_atlas2validsub_fix(i,2)=find(ind_validcell==ind_atlas2sub_fix(i,2));
        end
        
        % do Recog
        if(strcmpi(str_flagID,'noID'))
            [ind_atlas2validsub_pre,fid_tar2validsub_pre,fid_tar2sub_mean,fid_tar2sub_min]=RecogOnPosNoID(X_tar, X_sub, selected_pos_std_aff, ind_atlas2validsub_fix);%noGT
        elseif(strcmpi(str_flagID,'withID'))
            [ind_atlas2validsub_pre,fid_tar2validsub_pre,fid_tar2sub_mean,fid_tar2sub_min,rec_accuracy]=RecogOnPosNoID(X_tar, X_sub, selected_pos_std_aff, ind_atlas2validsub_fix, ind_atlas2validsub_man);%hasGT
        else
            fprintf('Invalid str_flagID:[%s], return!\n', str_flagID);
        end
        
%        rec_accuracy
%        output_rec_accuracy=['.\Accuracy\',filename_sub(1:end-length('.ano.ano.txt')),'Accuracy.txt'];
%        fprec_accuracy = fopen(output_rec_accuracy,'wt');
%        fprintf(fprec_accuracy, '%.8f,%.8f,%.8f,%.8f\n',rec_accuracy(1),rec_accuracy(2),rec_accuracy(3),rec_accuracy(4));
%        fclose(fprec_accuracy);
        
        fprintf('[5] Save recog results...\n');
        %[noGT]print recog result
        apo_sub=apo_sub_bk;
        %reinitialize apo_sub (remove previous name and comment)
        for i=1:length(apo_sub)
            cellname=strtrim(apo_sub{i}.name);
            apo_sub{i}.fidelity=0.0;
            if ~(strcmpi(cellname,'f1') || strcmpi(cellname,'f2') || strcmpi(cellname,'f3') || strcmpi(cellname,'f4') || strcmpi(cellname,'f5') || ...
                    strcmpi(cellname,'f6') || strcmpi(cellname,'f7') || strcmpi(cellname,'f8') || strcmpi(cellname,'f9') || strcmpi(cellname,'f10') || ...
                    ~isempty(strfind(cellname,'nouse')) || ~isempty(strfind(cellname,'NOUSE')))
                   % contains(cellname,'nouse') || contains(cellname,'NOUSE'))
                apo_sub{i}.name='';
                apo_sub{i}.comment='*AUTOFAIL*';
            end
        end
        %write recong ID and fidelity to apo_sub
        for i=1:length(apo_atlas)
            if(ind_atlas2validsub_pre(i)<1)%cannot find match
                continue;
            end
            ind_apo_sub=ind_validcell(ind_atlas2validsub_pre(i));
            apo_sub{ind_apo_sub}.name=strtrim(apo_atlas{i}.name);
            apo_sub{ind_apo_sub}.comment='*AUTO*';
            apo_sub{ind_apo_sub}.fidelity=fid_tar2validsub_pre(i);
        end
        %restore *FIX* comment to apo_sub
        for i=1:size(ind_atlas2sub_fix,1)
            apo_sub{ind_atlas2sub_fix(i,2)}.comment='*FIX*';
        end
        %save pre result to file 
        output_filename=[outpath,filename_sub(1:end-length('.ano.ano.txt')),'_recog.txt'];
        fp = fopen(output_filename, 'w');
		%[fp, message] = fopen(output_filename, 'w');
		%if fp < 0;
		% fprintf(2, 'failed to open "%s" because "%s"\n', fp, message);
		  %and here, get out gracefully
		%end
        for i=1:length(apo_sub)
            S=apo_sub{i};
            fprintf(fp, '%d,%s,%s,%s,%d,%d,%d,%.2f,%.2f,%.2f,%d,%d\n', ...
            S.n, ...
            strtrim(S.orderinfo), ...
            strtrim(upper(S.name)), ...
            strtrim(S.comment), ...
            S.z, ...
            S.x, ...
            S.y, ...
            S.pixmax, ...
            S.intensity, ...
            S.sdev, ...
            S.volsize, ...
            S.mass);           
        end
        fclose(fp);
        output_ano=[datapath,filename_sub(1:end-length('.ano.ano.txt')),'_recog.ano'];
        fp = fopen(output_ano,'wt');
        fprintf(fp, 'GRAYIMG=%s\n',[filename_sub(1:end-length('.ano.ano.txt')),'_crop_straight.raw']);
        fprintf(fp, 'MASKIMG=%s\n', [filename_sub(1:end-length('.ano.ano.txt')),'.ano.mask.raw']);
        fprintf(fp, 'ANOFILE=%s\n', [filename_sub(1:end-length('.ano.ano.txt')),'_recog.txt']);
        fclose(fp);
        output_filename=[outpath,filename_sub(1:end-length('.ano.ano.txt')),'_fidelity.txt'];
        fp = fopen(output_filename, 'w');
        for i=1:length(apo_sub)
            fidelity{i}=num2str(roundn(apo_sub{i}.fidelity,-4));
        end
        [~,ix]=sort(fidelity);
        apo_sub_sort=apo_sub(ix);
        determine_matrix=fid_tar2sub_mean{1};
        for i=1:size(determine_matrix,1)
            if(determine_matrix(i,4)<0.995)
                determine_matrix(i,2)=0;
                determine_matrix(i,3)=0;
                determine_matrix(i,4)=0;
            else
                determine_matrix(i,1)=0;
                determine_matrix(i,2)=0;
                determine_matrix(i,3)=0;
            end
        end
        [iii jjj]=find(determine_matrix==max(max(determine_matrix)));
%         fidelitySUM=0;
%         fidelityNUM=0;
%         fidelityMin=1;
%         for i=1:length(apo_sub_sort)
%             S=apo_sub_sort{i};
%             if ~(strcmpi(S.name,'f1') || strcmpi(S.name,'f2') || strcmpi(S.name,'f3') || strcmpi(S.name,'f4') || strcmpi(S.name,'f5') || ...
%                     strcmpi(S.name,'f6') || strcmpi(S.name,'f7') || strcmpi(S.name,'f8') || strcmpi(S.name,'f9') || strcmpi(S.name,'f10') || ...
%                     ~isempty(strfind(S.name,'nouse')) || ~isempty(strfind(S.name,'NOUSE'))||~isempty(strfind(upper(S.comment),'FIX')))
%             fidelitySUM=fidelitySUM+S.fidelity;
%             fidelityNUM=fidelityNUM+1;
%             fidelityMin=min(fidelityMin,S.fidelity);
%             end;
%         end;
            fprintf(fp,'#  final output, mean of fidelity=%.4f, min of fidelity=%.4f\n',fid_tar2sub_mean{1}(iii(length(iii)),jjj(length(jjj))),fid_tar2sub_min{1}(iii(length(iii)),jjj(length(jjj))));
%             for k=1:length(fid_tar2sub_mean)
%                 for i=1:size(fid_tar2sub_mean{k},1)
%                     for j=1:size(fid_tar2sub_mean{k},2)
%                         if (j==1)
%                             fprintf(fp,'#confidence score by atlas[%d], annotated by atlas[%d], rpm, mean of fidelity=%.10f, min of fidelity=%.10f\n',...
%                                 selected_atlas_ind(k),selected_atlas_ind(i),fid_tar2sub_mean{k}(i,j),fid_tar2sub_min{k}(i,j));
%                         else
%                             fprintf(fp,'#confidence score by atlas[%d], annotated by atlas[%d], bipartite[%d], mean of fidelity=%.10f, min of fidelity=%.10f\n',...
%                                 selected_atlas_ind(k),selected_atlas_ind(i),j-1,fid_tar2sub_mean{k}(i,j),fid_tar2sub_min{k}(i,j));
%                         end
%                     end
%                 end
%             end
            fprintf(fp,'# z, x, y, id, comment, fidelity\n');
        for i=1:length(apo_sub_sort)
            S=apo_sub_sort{i};
            if ~(strcmpi(S.name,'f1') || strcmpi(S.name,'f2') || strcmpi(S.name,'f3') || strcmpi(S.name,'f4') || strcmpi(S.name,'f5') || ...
                    strcmpi(S.name,'f6') || strcmpi(S.name,'f7') || strcmpi(S.name,'f8') || strcmpi(S.name,'f9') || strcmpi(S.name,'f10') || ...
                    ~isempty(strfind(S.name,'nouse')) || ~isempty(strfind(S.name,'NOUSE')))
                fprintf(fp, '%d,%d,%d,%s,%s,%.4f\n', ...
                S.z, ...
                S.x, ...
                S.y, ...
                strtrim(upper(S.name)), ...
                strtrim(S.comment), ...
                S.fidelity);
            end;
        end;
        fclose(fp);
        clear apo_sub_sort determine_matrix;
        fprintf('(%d):%d validcell recog done!\n',iter_file,nvalidcell);
        fprintf('Save predict result to [%s] done.\n', output_filename);
        
       if(strcmpi(str_flagID,'withID'))
            apo_sub=apo_sub_bk;
            %fill matching info matrix (in atlas order) man-vs-pre
            for i=1:558
                info_man_vs_pre{i,1}=strtrim(apo_atlas{i}.name);%manual name
                if(ind_atlas2validsub_pre(i)<1)
                    info_man_vs_pre{i,2}=-1;                           %apo index
                    info_man_vs_pre{i,3}='';                           %pre name
                    info_man_vs_pre{i,4}='---';                        %---:miss, xxx:wrong
                else    %找到当前atlas name被赋给了哪个cell并填充
                    ind_apo_sub=ind_validcell(ind_atlas2validsub_pre(i));%当前atlas name被赋给的那个cell的index
                    cell_manID=strtrim(apo_sub{ind_apo_sub}.name);%被赋予新name的cell原始manual name
                    for ii=1:558
                        if(strcmpi(cell_manID,strtrim(apo_atlas{ii}.name)))
                            info_man_vs_pre{ii,2}=ind_apo_sub;
                            info_man_vs_pre{ii,3}=strtrim(apo_atlas{i}.name);
                            info_man_vs_pre{ii,4}=fid_tar2validsub_pre(i);
                            break;
                        end
                    end
                    if(ind_atlas2validsub_man(i)~=ind_atlas2validsub_pre(i))
                        info_man_vs_pre{i,5}='xxx';
                    end
                end
            end
            %[hasGT]compute recog accu
            nmiss=558-nvalidcell;
            ind_validpre=find(ind_atlas2validsub_man>0);
            ncorrectpre=length(find(ind_atlas2validsub_man(ind_validpre)==ind_atlas2validsub_pre(ind_validpre)));
            nwrongpre=558-nmiss-ncorrectpre;
            recogaccu(iter_file)=ncorrectpre/558;
            fprintf('(%d):%d cell miss, %d/%d cell recog wrong, accuracy=(558-%d-%d)/558=%.4f\n', ...
                iter_file,nmiss,nwrongpre,nvalidcell,nmiss,nwrongpre,recogaccu(iter_file));
            %print recog analysis result to file (compare man and pre)
            %same as atlas cell order
            output_filename=[outpath,filename_sub(1:end-length('.ano.ano.txt')),'_recogreport.txt'];
            fp = fopen(output_filename, 'w');
            fprintf(fp,'[%d]:%d cell miss, %d/%d cell recog wrong, accuracy=(558-%d-%d)/558=%.4f\n', ...
                iter_file,nmiss,nwrongpre,nvalidcell,nmiss,nwrongpre,recogaccu(iter_file));
%             for k=1:length(fid_tar2sub_mean)
% 				for i=1:size(fid_tar2sub_mean{k},1)
% 				    for j=1:size(fid_tar2sub_mean{k},2)
% 					    if (j==1)
% 						    fprintf(fp,'#confidence score by atlas[%d], annotated by atlas[%d], rpm, mean of fidelity=%.10f, min of fidelity=%.10f, accuracy=%.4f\n',...
% 							    selected_atlas_ind(k),selected_atlas_ind(i),fid_tar2sub_mean{k}(i,1),fid_tar2sub_min{k}(i,1),rec_accuracy(i,1));
% 						else
% 							fprintf(fp,'#confidence score by atlas[%d], annotated by atlas[%d], bipartite[%d], mean of fidelity=%.10f, min of fidelity=%.10f, accuracy=%.4f\n',...
% 							    selected_atlas_ind(k),selected_atlas_ind(i),j-1,fid_tar2sub_mean{k}(i,j),fid_tar2sub_min{k}(i,j),rec_accuracy(i,j));
% 						end
% 					end
%                 end
%             end
            fprintf(fp,'#no, z, x, y, id_manual, id_auto, auto_confidence_score, errflag\n');
            for i=1:length(apo_atlas)
                if(ind_atlas2validsub_pre(i)<1)%cannot find match
                  fprintf(fp, '%d,  ,  ,  , %s, , ooooooooooooooo->Can not find match!\n', i, strtrim(apo_atlas{i}.name)); 
                  continue;
                end
                if(ind_atlas2validsub_man(i)<1)
                  fprintf(fp, '%d,  ,  ,  , %s, , mmmmmmmmmmmmmmm->This is the real missed cell!\n', i, strtrim(apo_atlas{i}.name)); 
                  continue;
                end
                errflag='';
                if(length(info_man_vs_pre{i,2})==0)
                    fprintf(fp, '%d,  ,  ,  , %s, , ooooooooooooooo->Can not find match!\n', i, strtrim(apo_atlas{i}.name));
                    continue;
                end
                if(ind_atlas2validsub_man(i)~=ind_atlas2validsub_pre(i)) errflag='xxxxxxxxxxxxxxx'; end
                fprintf(fp, '%d, %5.3f, %5.3f, %5.3f, %s, %s, %.4f, %s\n', i, ...
                    apo_sub{info_man_vs_pre{i,2}}.z, apo_sub{info_man_vs_pre{i,2}}.x, apo_sub{info_man_vs_pre{i,2}}.y, ...
                    info_man_vs_pre{i,1}, info_man_vs_pre{i,3},info_man_vs_pre{i,4}, errflag);
            end
            fclose(fp);
            fprintf('Save analysis result to [%s] done.\n', output_filename);
            clear info_man_vs_pre
        end
    end
%	end
end


function [ind_tar2sub,fid_tar2sub,fid_tar2sub_mean_all,fid_tar2sub_min_all,rec_accuracy_all]=RecogOnPosNoID(X_tar_all, X_sub, pos_std_aff_all, ind_tar2sub_fix, ind_tar2sub_man)
    X_sub_bk=X_sub;
    for atlas_ind=1:length(X_tar_all)
	%for atlas_ind=1:1
        X_sub=X_sub_bk;
        X_tar=X_tar_all{atlas_ind};
        pos_std_aff=pos_std_aff_all{atlas_ind};
        ntarcell=length(X_tar);
        nsubcell=length(X_sub);
        if nsubcell>558
            ind_tar2sub=-1*ones(nsubcell,1);
            ind_tar2sub_rpmout=-1*ones(nsubcell,1);
        else
            ind_tar2sub=-1*ones(ntarcell,1);%atlas(i)-->sub(ind_tar2sub(i))
            ind_tar2sub_rpmout=-1*ones(ntarcell,1);
        end
        fid_tar2sub=-1*ones(ntarcell,1);
        X_tar_bk=X_tar; 

        %whether groundtruth provided (manual anno for accu cpt)
        b_hasGT=1;
        if nargin<5
            b_hasGT=0;
        end

        %----------------------------------------------------------------------
        fprintf('\t(1) Do PCA alignment...\n');
        % normaliza point sets
        [X_tar, ~]=normalize_points(X_tar);
        [X_sub, ~]=normalize_points(X_sub);
        % PCA align
        tmp=cov(X_tar'); [~,~,V]=svd(tmp); T_tar=V;
        tmp=cov(X_sub'); [~,~,V]=svd(tmp); T_sub=V;
        T_pca=inv(T_tar')*T_sub'
        X_sub=T_pca*X_sub;
        clear V tmp T_tar T_sub T_pca
        %rectify mirror (mirror along 4 dirs and find dir with min dis)
        x=X_sub'; x_bk=x;
        y=X_tar';
        [xmax, dim] = size(x);
        [ymax, dim] = size(y);
        theta = [0,0,0; 0,0,180; 0,180,0; 0,180,180];
        for i=1:4
            sita_x=theta(i,1)/180*pi;  sita_y=theta(i,2)/180*pi;  sita_z=theta(i,3)/180*pi;
            Rx=[1, 0, 0; 0, cos(sita_x), sin(sita_x); 0, -sin(sita_x), cos(sita_x)];
            Ry=[cos(sita_y), 0, -sin(sita_y); 0, 1, 0; sin(sita_y), 0, cos(sita_y)];
            Rz=[cos(sita_z), sin(sita_z), 0; -sin(sita_z), cos(sita_z), 0; 0, 0, 1];
            x=x_bk*Rz*Ry*Rx;
            %记录各情况两点集接近程度tmp(i,j)=dis(y(i),x(j))^2
            tmp = zeros (ymax, xmax);
            for j=1:dim
                tmp = tmp + (y(:,j) * ones(1,xmax) - ones(ymax,1) * x(:,j)').^2;
            end
            near=sort(tmp,2);%sort each rows
            dis(i)=sum(sum(near(:,1:5)));%最近的5个点距离求和
        end
        [q,p]=sort(dis);
        sita_x=theta(p(1),1)/180*pi;  sita_y=theta(p(1),2)/180*pi;  sita_z=theta(p(1),3)/180*pi;
        Rx=[1, 0, 0; 0, cos(sita_x), sin(sita_x); 0, -sin(sita_x), cos(sita_x)];
        Ry=[cos(sita_y), 0, -sin(sita_y); 0, 1, 0; sin(sita_y), 0, cos(sita_y)];
        Rz=[cos(sita_z), sin(sita_z), 0; -sin(sita_z), cos(sita_z), 0; 0, 0, 1];
        x=x_bk*Rz*Ry*Rx;
        X_sub=x';
        clear i j x y x_bk xmax ymax dim theta i j sita_x sita_y sita_z tmp p q Rx Ry Rz near dis

        %     %plot cell point
        %     figure;
        %     plot3(X_tar(1,:),X_tar(2,:),X_tar(3,:),'r+','markersize',3); hold on
        %     plot3(X_sub(1,:),X_sub(2,:),X_sub(3,:),'bo','markersize',3);
        %     for i=1:length(X_tar)
        %         if(ind_tar2sub_man(i)<=0)
        %             plot3(X_tar(1,i),X_tar(2,i),X_tar(3,i),'ms','markersize',6,'markerfacecolor','m');
        %         else
        %             plot3([X_tar(1,i);X_sub(1,ind_tar2sub_man(i))],...
        %                 [X_tar(2,i);X_sub(2,ind_tar2sub_man(i))],...
        %                 [X_tar(3,i);X_sub(3,ind_tar2sub_man(i))]);
        %         end
        %     end
        %     xlabel('x'),ylabel('y'),zlabel('z')
        %     axis('equal'); grid on; set (gca, 'box', 'on');
        %     hold off;

        %----------------------------------------------------------------------
        fprintf('\t(2) Do RPM matching...\n');
        frac        = 1;
        T_init      = 0.006;%0.006
        T_final     = 0.0005;
        lamda1_init = 0.1;  %0.1 big=affine
        lamda2_init = 0.01; %0.01
        disp_flag   = 0;
        %     [c,d,vx,m]=cMIX_tps (ind_atlas2sub_man,X_sub',X_tar',frac,T_init,T_final,lamda1_init,lamda2_init,disp_flag);
        [c,d,vx,m]=cMIX_tps (X_sub',X_tar',frac,T_init,T_final,lamda1_init,lamda2_init,disp_flag);
        clear frac T_init T_final lamda1_init lamda2_init disp_flag c d vx
        mbk=m;
        %generate RPM matching index for bartite iteration
        for i=1:nsubcell
            [maxprob,ind_row]=max(m);%find max in each col
            [~,ind]=max(maxprob);
            ind_tar2sub(ind)=ind_row(ind);
            m(:,ind)=-1; m(ind_row(ind),:)=-1;
        end
        clear i maxprob ind_row ind m
        % load('tmp.mat');
        %generate fix matching RPM index for final RPM output
        m=mbk;
        for i=1:size(ind_tar2sub_fix,1)
            ind_tar2sub_rpmout(ind_tar2sub_fix(i,1))=ind_tar2sub_fix(i,2);
            m(:,ind_tar2sub_fix(i,1))=-1;m(ind_tar2sub_fix(i,2),:)=-1;
        end
        %generate matching RPM index for final RPM output
        for i=1:nsubcell
            [maxprob,ind_row]=max(m);%find max in each col
            [~,ind]=max(maxprob);
            if size(ind_tar2sub_fix,1)>0
                if ismember(ind_row(ind),ind_tar2sub_fix(:,2))
                else
                    ind_tar2sub_rpmout(ind)=ind_row(ind);
                end
            else
                ind_tar2sub_rpmout(ind)=ind_row(ind);
            end
            m(:,ind)=-1; m(ind_row(ind),:)=-1;
        end
        clear i maxprob ind_row ind m mbk ind_tar

        if b_hasGT
            rec_accuracy(1)=length(find(ind_tar2sub_man==ind_tar2sub_rpmout))/ntarcell;
        end
        ind_tar2sub_bk{1}=ind_tar2sub_rpmout;
        %----------------------------------------------------------------------
        % load('tmp.mat');
        for iter=1:3
            fprintf('\t(3) Affine align sub to atlas according to RPM matching result...\n');
            ind_tar2sub_valid=find(ind_tar2sub>0);
            X_tar=X_tar_bk(:,ind_tar2sub_valid);
            X_sub=X_sub_bk(:,ind_tar2sub(ind_tar2sub_valid));%reorder according to current matching result
            %update ind_tar2sub_fix according to current reorder

            T=affine3D_model(X_sub,X_tar);  %T*X_sub=X_tar
            X_sub2tar=T*[X_sub(1:3,:);ones(1,size(X_sub,2))];
            X_sub=X_sub2tar(1:3,:);
            clear T X_sub2tar

            %         figure;
            %         plot3(X_tar(1,:),X_tar(2,:),X_tar(3,:),'r+','markersize',3); hold on
            %         plot3(X_sub(1,:),X_sub(2,:),X_sub(3,:),'bo','markersize',3);
            %         for i=1:length(X_tar)
            %             plot3([X_tar(1,i);X_sub(1,i)],...
            %                   [X_tar(2,i);X_sub(2,i)],...
            %                   [X_tar(3,i);X_sub(3,i)]);
            %         end
            %         xlabel('x'),ylabel('y'),zlabel('z')
            %         axis('equal'); grid on; set (gca, 'box', 'on');
            %         hold off;

            %piece-wise affine warp the sub to atlas
            %so that the uneven stretch along worm can be better corrected
            fprintf('\t(4) Piece-wise affine warp the sub to atlas ...\n');
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
            for i=1:size(X_sub,2)
                cellarr_avg{i}=[];
            end
            for i=1:length(cellarr_ind_piece)
                for j=1:length(cellarr_ind_piece{i})
                    ind=cellarr_ind_piece{i}(j);
                    cellarr_avg{ind}=[cellarr_avg{ind},cellarr_X_sub2tar_piece{i}(:,j)];
                end
            end
            for i=1:size(X_sub,2)
                X_sub2tar_avg(:,i)=mean(cellarr_avg{i},2);
            end
            X_sub=X_sub2tar_avg;

            %         figure;
            %         plot3(X_tar(1,:),X_tar(2,:),X_tar(3,:),'r+','markersize',3); hold on
            %         plot3(X_sub(1,:),X_sub(2,:),X_sub(3,:),'bo','markersize',3);
            %         for i=1:length(X_tar)
            %             plot3([X_tar(1,i);X_sub(1,i)],...
            %                   [X_tar(2,i);X_sub(2,i)],...
            %                   [X_tar(3,i);X_sub(3,i)]);
            %         end
            %         xlabel('x'),ylabel('y'),zlabel('z')
            %         axis('equal'); grid on; set (gca, 'box', 'on');
            %         hold off;

            clear i xmin xmax piecesize piecestep npiece step
            clear cellarr_avg cellarr_ind_piece cellarr_X_sub2tar_piece ind j X_sub2tar_avg X_sub2tar_piece
            clear X_tar_piece xmax_piece xmin_piece T X_sub_piece

            fprintf('\t(5) Do bipartite cell recog based on cell relative pos and variation std ...\n');
            %load trained cell pos std
            %load(atlasmat,'pos_std_aff');
            arr_pos_var_pwaff=pos_std_aff.^2;
%             if iter==1
%                 fid_tar2sub_bk{1}=exp(-0.5*sum((X_tar-X_sub).^2./(20^2*arr_pos_var_pwaff(:,ind_tar2sub_valid))));
%                 fid_tar2sub_mean(1)=mean(exp(-0.5*sum((X_tar-X_sub).^2./(20^2*arr_pos_var_pwaff(:,ind_tar2sub_valid)))));
%             end
            % calculate the assignment energy of each atlas point to all topre points
            shape_prob=zeros(ntarcell,size(X_sub,2));%row:assignment energy of one atlas point to all topre points
            for i=1:ntarcell
                xyzdiff_atlas2topre=repmat(X_tar_bk(:,i),1,size(X_sub,2))-X_sub;
                exponential_term=0.5*sum(xyzdiff_atlas2topre.^2./(20^2*repmat(arr_pos_var_pwaff(:,i),1,size(X_sub,2))));
                shape_prob(i,:)=-exponential_term;
            end
            shape_prob=exp(shape_prob);
            clear i arr_pos_var_pwaff xyzdiff_atlas2topre exponential_term
            % assign a big energy to fixed cell pairs (so that they can get fixed during optimization)
            for i=1:size(ind_tar2sub_fix,1)
                ind_tar=ind_tar2sub_fix(i,1);%第i个sub被固定到的atlas ind
                ind_sub=find(ind_tar2sub==ind_tar2sub_fix(i,2));%固定的第i个sub在前次识别中匹配的atlas ind
                ind_sub_valid=find(ind_tar2sub_valid==ind_sub);%固定的第i个sub在当前重组的X_sub中的ind
                shape_prob(ind_tar,ind_sub_valid)=10;%max(shape_prob=1.0)
            end

            % do bipartite
            [mat_assignment,assigncost]=munkres(-shape_prob);
            % find atlas to pre matching index
            ind_tar2sub_bi=-1*ones(ntarcell,1);
            for i=1:ntarcell
                assignment=find(mat_assignment(i,:)==1);
                if(~isempty(assignment))
                    ind_tar2sub_bi(i)=ind_tar2sub(ind_tar2sub_valid(assignment));
                    %fid_tar2sub(i)=shape_prob(i,assignment);
                end
            end
            ind_tar2sub=ind_tar2sub_bi;
            if nsubcell>558
                ind_tar2sub(559:nsubcell)=-ones(length(559:nsubcell),1);
            end
            ind_tar2sub_bk{iter+1}=ind_tar2sub;
            %fid_tar2sub_bk{iter+1}=fid_tar2sub;
            %fid_tar2sub_mean(iter+1)=(sum(fid_tar2sub(find(fid_tar2sub<10)))+length(find(fid_tar2sub==10)))/length(fid_tar2sub);
            clear i mat_assignment assigncost ind_tar2sub_bi

            if b_hasGT
                rec_accuracy(1+iter)=length(find(ind_tar2sub_man==ind_tar2sub))/ntarcell;
            end

            clear i assignment
        end
           % fid_tar2sub_all{atlas_ind}=fid_tar2sub_bk;
        ind_tar2sub_all{atlas_ind}=ind_tar2sub_bk;
        %fid_tar2sub_mean_all(atlas_ind,:)=fid_tar2sub_mean;
        if b_hasGT
            rec_accuracy_all(atlas_ind,:)=rec_accuracy;
        end
    end
    fprintf('calculate fidelity score, ...\n');
	%for atlas_ind=2
	for atlas_ind=1:length(X_tar_all)
    	for iii=1:length(ind_tar2sub_all)
        	for jjj=1:length(ind_tar2sub_all{iii})
            	ind_tar2sub_valid=find(ind_tar2sub_all{iii}{jjj}>0);
            	X_tar=X_tar_all{atlas_ind}(:,ind_tar2sub_valid);
            	X_sub=X_sub_bk(:,ind_tar2sub_all{iii}{jjj}(ind_tar2sub_valid));
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
            	ncell=size(X_tar,2);
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
            	exponential_term=0.5*sum(xyzdiff_Xsub2tar.^2./((20*pos_std_aff_all{atlas_ind}(:,ind_tar2sub_valid)).^2));
            	shape_prob_pwaff_one=exp(-exponential_term);
                for i=1:size(ind_tar2sub_fix,1)
                    ind_sub_valid=find(ind_tar2sub_valid==ind_tar2sub_fix(i,1));
                    shape_prob_pwaff_one(ind_sub_valid)=1;% assign 1 as confidence score to fixed cells
                end
            	mean_shape_prob_pwaff{atlas_ind}(iii,jjj)=mean(shape_prob_pwaff_one);
            	min_shape_prob_pwaff{atlas_ind}(iii,jjj)=min(shape_prob_pwaff_one);
                shape_prob_pwaff_mis=shape_prob_pwaff_one;
                for i=1:length(ind_tar2sub_valid)
                    shape_prob_pwaff_one(ind_tar2sub_valid(i))=shape_prob_pwaff_mis(i);
                end
                allfididx=1:558;
                existidx = ismember(allfididx, ind_tar2sub_valid);
                fidmissidx=allfididx(~existidx); 
                for i=1:length(fidmissidx)
                    shape_prob_pwaff_one(fidmissidx(i))=-1;% assign -1 to missing cells 
                end
                shape_prob_pwaff{atlas_ind}{iii}{jjj}=shape_prob_pwaff_one;
            	clear X_sub X_tar step apo_tar apo_sub xmin xmax piecesize piecestep npiece xmin_piece xmax_piece ind X_sub_piece X_tar_piece T X_sub2tar_piece cellarr_X_sub2tar_piece cellarr_ind_piece cellarr_avg X_sub2tar_avg xyzdiff_Xsub2tar exponential_term;
            
        	end
    	end
    end
	%generate output by confidence score
    determine_matrix=mean_shape_prob_pwaff{1};
	for i=1:size(determine_matrix,1)
		if(determine_matrix(i,4)<0.995)
			determine_matrix(i,2)=0;
			determine_matrix(i,3)=0;
			determine_matrix(i,4)=0;
		else
			determine_matrix(i,1)=0;
			determine_matrix(i,2)=0;
			determine_matrix(i,3)=0;
		end
	end	
    [iii jjj]=find(determine_matrix==max(max(determine_matrix)));
    %iii=1;jjj=1;%for RMP output testing
    ind_tar2sub=ind_tar2sub_all{iii(length(iii))}{jjj(length(jjj))};
    fid_tar2sub=shape_prob_pwaff{1}{iii(length(iii))}{jjj(length(jjj))};
    fid_tar2sub_mean_all=mean_shape_prob_pwaff;
    fid_tar2sub_min_all=min_shape_prob_pwaff;
end


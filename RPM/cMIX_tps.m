function [o1,o2,o3,o4] =  cMIX_tps (in1,in2,in3,in4,in5,in6,in7,in8)
% function [o1,o2,o3,o4] =  cMIX_tps (apo_topre_reorder_ind,in1,in2,in3,in4,in5,in6,in7,in8)

if in8
    h=figure;
end
% fig(1); clf; whitebg('k'); set(gcf,'color',[0 0 0]);
% set(gcf,'DoubleBuffer','on')

% Init control parameters:
perT_maxit  = 10;
anneal_rate = 0.95;

lamda1_init = in6;
lamda2_init = in7;
disp_flag = in8;
x       = in1;
y       = in2;
frac	= in3;
T_init	= in4;
T_final = in5;
trans_type = 'tps';
sigma     = 1;
z         = x;

% init x,y,z:
[xmax, dim] = size(x); x = x (1:frac:xmax, :); [xmax, dim] = size(x);
[ymax, tmp] = size(y); y = y (1:frac:ymax, :); [ymax, tmp] = size(y);
[zmax, tmp] = size(z);
z = x;

% init m:
m              = ones (xmax, ymax) ./ (xmax * ymax);
T0             = max(x(:,1))^2;
moutlier       = 1/sqrt(T0)*exp(-1);       % /xmax *0.001;
% moutlier       = 1/xmax*0.01;
m_outliers_row = ones (1,ymax) * moutlier;
m_outliers_col = ones (xmax,1) * moutlier;

% init transformation parameters:
theta = 0; t = zeros (2,1); s = 1;
c_tps = zeros (xmax,dim+1);
d_tps = eye   (dim+1, dim+1);
w     = zeros (xmax+dim+1, dim);

% -------------------------------------------------------------------
% Annealing procedure:
% -------------------------------------------------------------------
T       = T_init;

vx = x;
vy = y;

it_total = 1;
flag_stop = 0;
while (flag_stop ~= 1)
    for i=1:perT_maxit     % repeat at each termperature.
        % Given vx, y, Update m:
        m = cMIX_calc_m (vx, y, T, m_outliers_row, m_outliers_col,it_total);
        
        % Given m, update transformation:
        vy = m * y ./ ( (sum(m'))' * ones(1,dim));
        
        lamda1 = lamda1_init*length(x)*T;
        lamda2 = lamda2_init*length(x)*T;
        
        [c_tps, d_tps, w] = cMIX_calc_transformation (trans_type, ...
            lamda1, lamda2, sigma, x, vy, z);
        [vx] = cMIX_warp_pts (trans_type, x, z, c_tps, d_tps, w, sigma);
    end  % end of iteration/perT
    
    T = T * anneal_rate;
    
    % Determine if it's time to stop:
%     fprintf ('T = %.4f:\t lamda1: %.4f lamda2: %.4f\n', T, lamda1, lamda2);
    if T < T_final; flag_stop = 1; end;
    
    % Display:
    if disp_flag
        it_total = it_total + 1;
        figure(h)
        plot3(y(:,1),y(:,2),y(:,3),'r+','markersize',3); hold on
        plot3(vx(:,1),vx(:,2),vx(:,3),'bo','markersize',3);
%         for i=1:length(y)
%             plot3([y(i,1);vx(apo_topre_reorder_ind(i),1)],...
%                 [y(i,2);vx(apo_topre_reorder_ind(i),2)],...
%                 [y(i,3);vx(apo_topre_reorder_ind(i),3)]);
%         end
        xlabel('x'),ylabel('y'),zlabel('z')
        axis('equal'); grid on; set (gca, 'box', 'on');
        
        hold off; drawnow;
    end
end % end of annealing.

o1 = c_tps;
o2 = d_tps;
o3 = vx;
o4 = m;
end

function [m, m_outliers_row, m_outliers_col] = cMIX_calc_m ...
    (vx, y, T, m_outliers_row, m_outliers_col, it_total)

[xmax,dim] = size(vx);
[ymax,dim] = size(y);

% Given v=tranformed(x), update m:
y_tmp = zeros (xmax, ymax);
for it_dim=1:dim
    y_tmp = y_tmp + (vx(:,it_dim) * ones(1,ymax) - ones(xmax,1) * y(:,it_dim)').^2;
end

m_tmp = 1/sqrt(T) .* exp (-y_tmp/T);
% m_tmp = m_tmp + randn(xmax, ymax) * (1/xmax) * 0.001;

m = m_tmp;

% normalize accross the outliers as well:
sy         = sum (m) + m_outliers_row;
m          = m ./ (ones(xmax,1) * sy);
end
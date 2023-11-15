
im = imread('images/roller_coaster.jpg');
load('images/roller_coaster_data.mat');
nrcyl = length(CCs);
Pgtn = K\Pgt;


%% reprojections using gt cameras
figure(1)
% clf
% imshow(im)
% hold on

gv = linspace(-30,30,50); % adjust for appropriate domain
[x, y, z]=meshgrid(gv, gv, gv);
axis equal

% figure

for jjj = 1:nrcyl
    CC = CCs(:,:,jjj);
    VP = VPs(:,jjj);
    [U,S,V] = svd(CC);
    S = S./10000000000000000000;
    U = U./U(16);
    U([4,8,12,13,14,15,16]) = zeros(1, 7);
    CC = U'*S*U;
    % CC = [1, 1, 1, 1; 1, 1, 0, 0; 1, 0, 1, 0; 1, 0, 0, 1];
    % CC = fliplr(fliplr(CC).');
    % F = CC(1) + CC(6)*x.*x + CC(11)*y.*y + CC(16)*z.*z + 2*CC(2)*x + 2*CC(3)*y + 2*CC(4)*z + 2*CC(7)*x.*y + 2*CC(8)*x.*z + 2*CC(12)*y.*z;
    % F = CC(16) + CC(1)*x.*x + CC(6)*y.*y + CC(11)*z.*z + 2*CC(13)*x + 2*CC(14)*y + 2*CC(15)*z + 2*CC(2)*x.*y + 2*CC(3)*x.*z + 2*CC(8)*y.*z
    % a = append(num2str(CC(16)), " + ", num2str(CC(1)), "*x^2 + ", num2str(CC(6)), "*y^2 + ", num2str(CC(11)), "*z^2 + ", num2str(2*CC(13)), "*x + ", num2str(2*CC(14)), "*y + ", num2str(2*CC(15)), "*z + ", num2str(2*CC(2)), "*x*y + ", num2str(2*CC(3)), "*x*z +", num2str(2*CC(8)), "*y*z", "=0");
    a = append(num2str(CC(1)), " + ", num2str(CC(6)), "*x^2 + ", num2str(CC(11)), "*y^2 + ", num2str(CC(16)), "*z^2 + ", num2str(2*CC(2)), "*x + ", num2str(2*CC(3)), "*y + ", num2str(2*CC(4)), "*z + ", num2str(2*CC(7)), "*x*y + ", num2str(2*CC(8)), "*x*z +", num2str(2*CC(12)), "*y*z", "=0");
    % a = append(num2str(VP(1)), "*x + ", num2str(VP(2)), "*y + ", num2str(VP(3)), "*z + ", num2str(VP(4)), " = 0");
    disp(a);
    disp(a);
    % F = sqrt(x.^2 + y.^2 + z.^2) - 10;
    % isosurface(x, y, z, F, 0);
    % [l1,l2] = linesfromquadric(CCs(:,:,jjj),VPs(:,jjj),Pgt);
    % rital(ll(:,2*jjj-1),'b-');
    % rital(l1,'r--');
    % rital(ll(:,2*jjj),'b-');
    % rital(l2,'r--');
end
legend({'Detected lines','Reprojected lines'}, 'FontSize',14)
title('Reprojections using ground truth cameras', 'FontSize',20)



%% test pose with focal

fe = 1000; % scale problem for numerical stability
Ke = K;
Ke(1) = fe;
Ke(5) = fe;

bnd = 0.01;
bnd2 = 0.005;
lille = 1e-9;


lln = Ke'*ll;
lln = lln./sqrt(sum(lln.^2));


vpds = [VPs(1:3,:);VPs(1:3,:)];
vpds = reshape(vpds,3,[]);

CDs = cell(1,2*nrcyl);
for iii = 1:nrcyl
    CDs{2*iii-1} = CCs(:,:,iii)/norm(CCs(:,:,iii));
    CDs{2*iii} = CCs(:,:,iii)/norm(CCs(:,:,iii));
end

warning('off');


ins = 0;
fbest = nan;

% test all quads of lines
cids = nchoosek(1:2*nrcyl,4);
for ci = 1:size(cids,1)
    if ~any(diff(cids(ci,:)')==1)
        
        
        l = lln(:,cids(ci,:));
        vpd = vpds(:,cids(ci,:));
        [R,f] = fullsolver_focalcylinderpose(l,vpd);
        for idi = 1:length(f)
            ins = [ins;sum(abs(diag(lln'*diag([f(idi) f(idi) 1])*reshape(R(:,idi),3,3)*vpds))<bnd)];
            if ins(end)>max(ins(1:end-1))
                Rbest = reshape(R(:,idi),3,3);
                fbest = f(idi);
                idsbest = cids(ci,:);
            end
            
        end
    end
end

ins2 = 0;
% check translation
Pe = Rbest;
K2 = diag([fbest fbest 1]);
lln2 =   K2'*lln;

alldata = zeros(10,2*nrcyl);
for jjj = 1:2*nrcyl
    alldata(:,jjj) = getdata_translation_cylinderpose(CDs{jjj},lln2(:,jjj),Pe)';
end

qbest = rot2quat(Rbest);
data = zeros(40,1);
data(10:13) = qbest;
% test all triplets of lines
cids = nchoosek(1:2*nrcyl,3);
for ci = 1:size(cids,1)
    
    data(1:9) = lln2(:,cids(ci,:));
    data(14:22) = -CDs{cids(ci,1)}([1 2 3 4 6 7 8 11 12])/CDs{cids(ci,1)}(16);
    data(23:31) = -CDs{cids(ci,2)}([1 2 3 4 6 7 8 11 12])/CDs{cids(ci,2)}(16);
    data(32:40) = -CDs{cids(ci,3)}([1 2 3 4 6 7 8 11 12])/CDs{cids(ci,3)}(16);
    
    %data = alldata(:,cids(ci,:));
    
    
    %sols = solver_translation_cylinderpose(data(:));
    sols = solver_conics_translation(data);
    sols = sols(:,sum(abs(imag(sols)))<lille);
    nsols = size(sols,2);
    for jjj = 1:nsols
        t1 = sols(1,jjj);
        t2 = sols(2,jjj);
        t3 = sols(3,jjj);
        vv = [t1^2, t1*t2, t1*t3, t1, t2^2, t2*t3, t2, t3^2, t3, 1].';
        res = alldata'*vv;
        insi = sum(abs(res)<bnd2);
        ins2 = [ins2;insi];
        if ins2(end)>max(ins2(1:end-1))
            tbest = [t1;t2;t3];
        end
    end
end
bestP = [Rbest tbest];
bestf = fbest*fe;

disp("Pgtn:")
disp(Pgtn)
disp("bestP:")
disp(bestP)

disp("K(1):")
disp(K(1))
disp("bestf:")
disp(bestf)


%% reprojections using estimated cameras
figure(2)
clf
Ke = K;
Ke(1) = bestf;
Ke(5) = bestf;

Pe = Ke*bestP;
imshow(im)
hold on


for jjj = 1:nrcyl
    [l1,l2] = linesfromquadric(CCs(:,:,jjj),VPs(:,jjj),Pe);
    rital(ll(:,2*jjj-1),'b-');
    rital(l1,'r--');
    rital(ll(:,2*jjj),'b-');
    rital(l2,'r--');
end
legend({'Detected lines','Reprojected lines'}, 'FontSize',14)
title('Reprojections using estimated cameras', 'FontSize',20)






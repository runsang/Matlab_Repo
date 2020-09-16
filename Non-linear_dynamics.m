clc
clear all
close all
%% Lorenz and Rossler systems
% lorenz: sigma=10,rho=28,beta=8/3
% rossler: a=0.1,b=0.1,c=14
sigma = 10;
rho = 28;
beta = 8/3;
par = [sigma,rho,beta];

dat0 = [0,1,0];
tspan = 0:0.001:100;
eps = 1e-5;
options = odeset('RelTol',eps,'AbsTol',[eps eps eps/10]);
[t,X] = ode45(@(t,x)lorenz(t,x,par),tspan,dat0,options);
% [t,X] = ode45(@(t,x)rossler(t,x,par),tspan,dat0,options);

% figure('color','w','Position',[645,492,390,301])
% plot3(X(:,1),X(:,2),X(:,3));
% % ttstr = strcat('Lorenz Attractor:',' sigma=',num2str(sigma),' rho=',num2str(rho),' beta=',num2str(beta));
% ttstr = strcat('Rossler Attractor:',' a=',num2str(sigma),' b=',num2str(rho),' c=',num2str(beta));
% title(ttstr,'FontSize',10,'FontWeight','bold');
% xlabel('X','FontSize',10,'FontWeight','bold');
% ylabel('Y','FontSize',10,'FontWeight','bold');
% zlabel('Z','FontSize',10,'FontWeight','bold');
% set(gca,'LineWidth',2,'FontSize',10,'FontWeight','bold');
% view([-104,23])
%% Logistic Growth
% scale = 10000; 
% maxpoints = 200; 
% N = 3000; 
% rs = linspace(2.5,5,N); 
% M = 500; 
% 
% for j = 1:length(rs) 
%     r=rs(j); 
%     x=zeros(M,1); 
%     x(1) = 0.1; 
%     
%     for i = 2:M
%         x(i) = r*x(i-1)*(1-x(i-1));
%     end
%    
%     out{j} = unique(round(scale*x(end-maxpoints:end)));
% end
% % Rearrange cell array into a large n-by-2 vector for plotting
% data = [];
% for k = 1:length(rs)
%     n = length(out{k});
%     data = [data;  [rs(k)*ones(n,1),out{k}]];
% end
% % Plot the data
% figure('color','w')
% h=plot(data(:,1),data(:,2)/scale,'.','Color',[0.9100 0.4100 0.1700],'MarkerSize',1);
% xlabel('r')
% ylabel('Attractor')
% set(gca,'LineWidth',2,'FontSize',10,'FontWeight','bold');
%% Mutual Information
partitions = 10;
tau = 20;
len = length(X);
mi = mutual_info(X(:,1),partitions,tau);

figure('Position',[100 400 460 360]);
plot(0:1:tau,mi,'o-','MarkerSize',5);
title('Mutual Information Test (first local minimum)','FontSize',10,'FontWeight','bold');
xlabel('Delay (sampling time)','FontSize',10,'FontWeight','bold');
ylabel('Mutual Information','FontSize',10,'FontWeight','bold');
get(gcf,'CurrentAxes');
set(gca,'FontSize',10,'FontWeight','bold');
grid on;
%% FNN
mindim = 1;
maxdim = 10;
tau = 8;
rt = 10;
eps0 = 1/1000;
out = false_nearest(X(:,1),mindim,maxdim,tau,rt,eps0);

fnn = out(:,1:2);
figure('Position',[100 400 460 360]);
plt=plot(fnn(:,1),fnn(:,2),'o-','MarkerSize',4.5);
title('False nearest neighbor test','FontSize',10,'FontWeight','bold');
xlabel('dimension','FontSize',10,'FontWeight','bold');
ylabel('FNN','FontSize',10,'FontWeight','bold');
get(gcf,'CurrentAxes');
set(gca,'LineWidth',2,'FontSize',10,'FontWeight','bold');
grid on;
%% Utility
function dxyz = lorenz(t,dat,par)
x = dat(1);
y = dat(2);
z = dat(3);
sigma = par(1);
rho = par(2);
beta = par(3);

dx = sigma*(y-x);
dy = rho*x-y-x*z;
dz = x*y-beta*z;

dxyz = [dx;dy;dz];
end

function dxyz = rossler(t,dat,par)
x = dat(1);
y = dat(2);
z = dat(3);
a = par(1);
b = par(2);
c = par(3);

dx = -y-z;
dy = x+a*y;
dz = b+z*(x-c);

dxyz = [dx;dy;dz];
end

function mi = mutual_info(data,partitions,tau)
% time delayed mutual information
av = mean(data);
variance = var(data);
minimum = min(data);
maximum = max(data);
interval = maximum-minimum;
len = length(data);
% normalize between 0 and 1
for i = 1:1:len
    data(i) =(data(i)- minimum)/interval;
end

for i = 1:1:len
    if data(i) > 0 
        array(i) = ceil(data(i)*partitions);
    else
        array(i) = 1;
    end
end

for i = 0:1:tau
    mi = make_cond_entropy(i,array,len,partitions);
end

end

function mi = make_cond_entropy(t,array,len,partitions)
hi=0;
hii=0;
count=0;
hpi=0;
hpj=0;
pij=0;
cond_ent=0.0;
h2 = zeros(partitions,partitions);
for i = 1:1:partitions
    h1(i)=0;
    h11(i)=0;
end
for i=1:1:len
    if i > t
        hii = array(i);
        hi = array(i-t);
        h1(hi) = h1(hi)+1;
        h11(hii) = h11(hii)+1;
        h2(hi,hii) = h2(hi,hii)+1;
        count = count+1;
    end
end
norm=1.0/double(count);
cond_ent=0.0;
for i=1:1:partitions
    hpi = double(h1(i))*norm;
    if hpi > 0.0
        for j = 1:1:partitions
            hpj = double(h11(j))*norm;
            if hpj > 0.0
                pij = double(h2(i,j))*norm;
                if (pij > 0.0)
                    cond_ent = cond_ent + pij*log(pij/hpj/hpi);
                end
            end
        end
    end
end
mi = cond_ent;
end

function out = false_nearest(signal,mindim,maxdim,tau,rt,eps0)
%Determines the fraction of false nearest neighbors.
%Author: Hui Yang
%Affiliation: 
       %The Pennsylvania State University
       %310 Leohard Building, University Park, PA
       %Email: yanghui@gmail.com
%signal: input time series     
%mindim - minimal dimension of the delay vectors 	1
%maxdim - maximal dimension of the delay vectors 	5
%tau - delay of the vectors 	1
%rt - ratio factor 	10.0
% If you find this demo useful, please cite the following paper:
% [1]	H. Yang, �Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) 
% Signals,� IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
% DOI: 10.1109/TBME.2010.2063704
% [2]	Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and 
% nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012 
% DOI: 10.1016/j.chaos.2012.03.013
if nargin<2 | isempty(mindim)
  mindim = 1;
end
if nargin<3 | isempty(maxdim)
  maxdim = 5;
end
if nargin<4 | isempty(tau)
  tau = 1;
end
if nargin<5 | isempty(rt)
  rt = 10;
end
if nargin<6 | isempty(eps0)
  eps0=1/1000;
end
minimum = min(signal);
maximum = max(signal);
interval = maximum-minimum;
len = length(signal);
BOX = 1024;
ibox = BOX-1;
theiler = 0;
global aveps vareps variance box list toolarge
% data is normalized
for i = 1:1:len
    signal(i) =(signal(i)- minimum)/interval;
end
av = mean(signal);
variance = std(signal);
out = zeros(maxdim,4);
for dim = mindim:maxdim
    epsilon=eps0;
    toolarge=0;
    alldone=0;
    donesofar=0;
    aveps=0.0;
    vareps=0.0;
    
    for i=1:len
      nearest(i)=0;
    end
    
    fprintf('Start for dimension=%d\n',dim);
    
    while (~alldone && (epsilon < 2*variance/rt)) 
        alldone=1;
        make_box(signal,len-1,dim,tau,epsilon);
        for i=(dim-1)*tau+1:(len-1)
            if (~nearest(i))
                nearest(i)=find_nearest(i,dim,tau,epsilon,signal,rt,theiler);
                alldone = bitand(alldone,nearest(i));
                donesofar = donesofar+nearest(i);
            end
        end
        
        fprintf('Found %d up to epsilon=%d\n',donesofar,epsilon*interval);
        
        epsilon=epsilon*sqrt(2.0);
        if (~donesofar)
            eps0=epsilon;
        end
    end
    if (donesofar == 0)
      fprintf('Not enough points found!\n');
      fnn = 0;
    else
        aveps = aveps*(1/donesofar);
        vareps = vareps*(1/donesofar);
        fnn = toolarge/donesofar;
    end
    out(dim,:) = [dim fnn aveps vareps];
    
end
end  
  
function y = find_nearest(n,dim,tau,eps,signal,rt,theiler)
global aveps vareps variance box list toolarge
element=0;
which= -1;
dx=0;
maxdx=0;
mindx=1.1;
factor=0;
ibox=1023;
x=bitand(ceil(signal(n-(dim-1)*tau)/eps),ibox);
if x==0
    x=1;
end
y=bitand(ceil(signal(n)/eps),ibox);
if y==0
    y=1;
end
for x1=x-1:x+1
    if x1==0
        continue
    end
    x2= bitand(x1,ibox);
    for y1=y-1:y+1
        if y1==0
            continue
        end
        element = box(x2,bitand(y1,ibox));
        while (element ~= -1)
            if (abs(element-n) > theiler) 
                maxdx=abs(signal(n)-signal(element));
                for i=1:dim
                    i1=(i-1)*tau;
                    dx = abs(signal(n-i1)-signal(element-i1));
                    if (dx > maxdx)
                        maxdx=dx;
                    end
                end
                if ((maxdx < mindx) && (maxdx > 0.0))
                    which = element;
                    mindx = maxdx;
                end
            end
            element = list(element);
        end
    end
end
if ((which ~= -1) && (mindx <= eps) && (mindx <= variance/rt)) 
    aveps = aveps+mindx;
    vareps = vareps+mindx*mindx;
    factor=abs(signal(n+1)-signal(which+1))/mindx;
    if (factor > rt)
      toolarge=toolarge+1;
    end
    y = 1;
else
    y = 0;
end
end

function make_box(ser,l,dim,del,eps)
global box list
bs=1024;
ib=bs-1;
box = -ones(bs,bs);
  
for i=(dim-1)*del+1:l
    x = bitand(ceil(ser(i-(dim-1)*del)/eps),ib);
    if x==0
        x=1;
    end
    y = bitand(ceil(ser(i)/eps),ib);
    if y==0
        y=1;
    end
    list(i)=box(x,y);
    box(x,y)=i;
end

end

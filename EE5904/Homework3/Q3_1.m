%% EE5904 Part 1 Homemwork3 Q3_1
% Writen by An Lijun for EE5904

%init
close all;clear;clc;

t = linspace(-pi,pi,200);
trainX=[t.*sin(pi*sin(t)./t); 1-abs(t).*cos(pi*sin(t)./t)]; %2x200 matrix 

%parameter setting
w=rand(25,2); %randomly init weigth 40 neurons in output layer
sigma0=sqrt(1^2+25^2)/2; %M=1,N=40
eta0=0.1;
T=500; %iterations
tau1=T/log(sigma0);
tau2=T;
eta=eta0;
sigma=sigma0;

% training
for n=1:T
    i=randperm(size(trainX,2),1);%randomly select vector x
    %competitive process
    [min_dist,Idx]=min(dist(trainX(:,i)',w'));% 1*2 * 2*25 =1*25
    %adaptation process
    for j=1:25
       h=exp((j-Idx).^2/-(2*sigma.^2));
       w(j,:)=w(j,:)+eta*h*(trainX(:,i)'-w(j,:));
    end 
    %update eta&sigma
    eta=eta0*exp(-n/tau2);
    sigma=sigma0*exp(-n/tau1);
end
figure(1)
plot(trainX(1,:),trainX(2,:),'--','LineWidth',1.5);hold on;
plot(w(:,1),w(:,2),'LineWidth',1); hold on;
scatter(w(:,1),w(:,2),'o');hold on;
axis([-pi,pi,-2.5,2]);
title(['The topological adjacent neurons , Epoch=',num2str(T)]);
legend('ideal output','SOM output','neurons');
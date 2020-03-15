%% EE5904 Part 1 Homemwork3 Q3_2
% Writen by An Lijun for EE5904
%init
close all;clear;clc;

trainX = rands(2,500); 
%parameters setting

w=rand(2,5,5); %randomly init weigth 25 neurons in output layer
sigma0=sqrt(5^2+5^2)/2; %M=5 N=5
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
    distance=zeros(5,5);
    for row=1:5
        for col=1:5
            distance(row,col)=sqrt((trainX(1,i)-w(1,row,col)).^2+(trainX(2,i)-w(2,row,col)).^2);
        end
    end
    [min_row,min_col]=find(distance==min(min(distance)));
    %adaptation process
    for row=1:5
        for col=1:5
            h=exp(((row-min_row).^2+(col-min_col).^2)/-(2*sigma.^2));
            w(:,row,col)=w(:,row,col)+eta*h*(trainX(:,i)-w(:,row,col));
        end
    end
    %update eta&sigma
    eta=eta0*exp(-n/tau2);
    sigma=sigma0*exp(-n/tau1);
end

%plot
figure(1)
plot(trainX(1,:),trainX(2,:),'+r');hold on;
axis equal;
for i=1:5
    plot(w(1,:,i),w(2,:,i),'LineWidth',1.5);
    scatter(w(1,:,i),w(2,:,i),'o');
    hold on;    
end
for j=1:5
    w_1 = reshape(w(1,j,:),1,5);
    w_2 = reshape(w(2,j,:),1,5);
    plot(w_1,w_2,'LineWidth',1.5);
    hold on; 
end
title(['The topological adjacent neurons , Epoch=',num2str(T)]);
legend('ideal output','SOM output','neurons');
clc;
clear;
h=4;              %the node number of input layer
i=3;              %the node number of hidden layer
j=3;              %the node number of output layer
V=2 * (rand(h,i) - 0.5);           %the weight between inout and hidden layers
W=2 * (rand(i,j) - 0.5);           %the weight between hidden and output layers
Pi=2*(rand(1,i)-0.5);            %the thresholds of hidden layer nodes
Tau=2*(rand(1,j)-0.5);           %the thresholds of output layer nodes
a=ones(h,1);             %the input values
b=ones(i,1);             %the hidden layernode activations
c=ones(j,1);             %the output layer mode activations
ck=ones(j,1);            %the desired output of output layer nodes
d=ones(j,1);             %the error in output layer nodes
e=ones(i,1);             %the error in hidden layer nodes
DeltaW=zeros(i,j);       %the amount of change for the weights W[i,j]
MomentW=zeros(i,j);      
DeltaV=zeros(h,i);       %the amount of change for the weights V[i,j]
MomentV=zeros(h,i);      
DeltaPi=zeros(i,1);      %the amount of change for the thresholds Pi[i,j]
DeltaTau=zeros(j,1);     %the amount of change for the thresholds Tau[i,j]
Alpha = 0.1;             %the learning rate
Beta=0.1;                %the learning rate
Gamma=0.9;               
TOR=0.001;              %determines when to stop train
Maxepoch=10000;          %the max epoch

matdata = load('1800271013.mat');
%���������
%matdata.dataset = matdata.dataset(randperm(size(matdata.dataset,1))',:);
%��ȡ����ǩ��
temp_labels = matdata.dataset(:,2);
labels = zeros(150,3);
%����ǩ����λ����ʾ
for ii=1:size(temp_labels)
    if temp_labels(ii)==0
        labels(ii,:)=[1 0 0];
    elseif temp_labels(ii)==1
        labels(ii,:)=[0 1 0];
    else
        labels(ii,:)=[0 0 1];
    end
end
%��ȡ��������
input = matdata.dataset(:,3:6);


%��������׼������
%[input,minI, MaxI] = premnmx(input');
%input=input';

%����ѵ�����Ͳ��Լ���75%Ϊѵ������25%Ϊ���Լ�
[rows,cols]=size(input);
train_input=input(1:ceil(rows*0.75),:);
train_label=labels(1:ceil(rows*0.75),:);
test_input=input(ceil(rows*0.75)+1:rows , :);
test_label=labels(ceil(rows*0.75)+1:rows,:);

%��������׼������
[train_input, minI, maxI] = premnmx(train_input');
train_input=train_input';
test_input = tramnmx( test_input' , minI, maxI ) ;
test_input = test_input';

stop_train = 0;%loss������͵�ʱֹͣѵ��
ErrorHistory = [];
for ii = 1:Maxepoch
    if stop_train==1
        break;
    end
    for jj = 1:ceil(rows*0.75)
        b = logsig(train_input(jj,:) * V + Pi);
        ck = train_label(jj,:);
        c = logsig(b * W + Tau);
        
        E = 0.5*sumsqr(c - ck);
        
        if E < TOR
            stop_train = 1;
            break;
        end
        %����Fc�Ĵ���
        for FC_k = 1:j        
            d(FC_k) = c(FC_k)*(1-c(FC_k))*(ck(FC_k)-c(FC_k));
        end
        
        %����FB�Ĵ���
        for FB_k = 1:i
            error = 0;
            for FC_k = 1:j
                error = error+d(FC_k)*W(FB_k,FC_k);
            end
            e(FB_k)=b(FB_k)*(1-b(FB_k))*error;
        end
        %�޸�FC���FB֮���Ȩֵwij
        for FB_k = 1:i
            for FC_k = 1:j
                %MomentW(FB_k,FC_k) = Gamma*MomentW(FB_k,FC_k)-Alpha*b(FB_k)*d(FC_k);
                %W(FB_k,FC_k) = W(FB_k,FC_k)+MomentW(FB_k,FC_k);

                DeltaW(FB_k,FC_k)=Alpha*b(FB_k)*d(FC_k);
                W(FB_k,FC_k) = W(FB_k,FC_k)+DeltaW(FB_k,FC_k);
            end
        end

        
        %�޸�FA���FB��֮���Ȩֵ
        for FA_k = 1:h
            for FB_k = 1:i
                %MomentV(FA_k,FB_k)=Gamma*MomentV(FA_k,FB_k) + Beta*a(FA_k)*e(FB_k);
                %V(FA_k,FB_k) = V(FA_k,FB_k)+MomentV(FA_k,FB_k);

                DeltaV(FA_k,FB_k)=Beta*a(FA_k)*e(FB_k);
                V(FA_k,FB_k) = V(FA_k,FB_k)+DeltaV(FA_k,FB_k);
            end
        end
        %�޸�ƫ��
        for FB_k=1:i
            DeltaPi(FB_k)=Beta*e(FB_k);
            Pi(FB_k) = Pi(FB_k) + DeltaPi(FB_k);
        end
        for FC_k=1:j
            DeltaTau(FC_k)=Alpha*d(FC_k);
            Tau(FC_k) = Tau(FC_k)+DeltaTau(FC_k);
        end          
    end
    ErrorHistory = [ErrorHistory E];
end

[test_len, none] = size(test_input);
correct_count=0;
count=0;
for jj = 1:test_len
    b = logsig(test_input(jj,:) * V + Pi);
    ck = test_label(jj,:);
    c = logsig(b * W + Tau);
    [C,pred]=max(c);
    [CK,label]=max(ck);
    count = count +1;
    if label==pred
        correct_count = correct_count + 1;
    end    
end
acc = correct_count/count;

%�۲���������(���ƽ����)��ѵ������������еı仯���------------------------------------------
figure(3);

n = length(ErrorHistory);
t3 = 1:n;
plot(t3, ErrorHistory, 'r-');

%Ϊ�˸�������ع۲����������ֵ�ı仯�����������ֻ����ǰ100�ε�ѵ�����
xlim([1 n]);
xlabel('ѵ������');
ylabel('��������ֵ');
title('��������(���ƽ����)��ѵ������������еı仯ͼ');
grid on;









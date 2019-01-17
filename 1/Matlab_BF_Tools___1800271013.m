%��ȡѵ������                                   
matdata = load('1800271013.mat');

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

%����ѵ�����Ͳ��Լ���75%Ϊѵ������25%Ϊ���Լ�
[rows,cols]=size(input);
train_input=input(1:ceil(rows*0.75),:);
train_label=labels(1:ceil(rows*0.75),:);
test_input=input(ceil(rows*0.75)+1:rows , :);
test_label=labels(ceil(rows*0.75)+1:rows,:);


%��������׼������
[train_input, minI, maxI] = premnmx(train_input');
%train_input=train_input';
test_input = tramnmx( test_input' , minI, maxI ) ;
%test_input = test_input';

%����������
net = newff( minmax(train_input) , [3,3] , { 'logsig' 'logsig' } , 'traingdx' ) ; 

%����ѵ������
net.trainparam.show = 50 ;
net.trainparam.epochs = 500 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.005 ;

%��ʼѵ��
net = train( net, train_input , train_label' ) ;


%����
Y = sim( net , test_input ); 
%ͳ��ʶ����ȷ��
[s1 , s2] = size( Y ) ;
test_label = test_label';
hitNum = 0 ;
for i = 1 : s2
    [m , c_Index] = max( Y(:,i)) ;
    [n,ck_Index] = max(test_label(:,i));
    if( c_Index  == ck_Index) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('accuracy %3.3f%%',100 * hitNum / s2 )

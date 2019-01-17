%读取训练数据                                   
matdata = load('1800271013.mat');

%抽取出标签列
temp_labels = matdata.dataset(:,2);
labels = zeros(150,3);
%将标签用三位数表示
for ii=1:size(temp_labels)
    if temp_labels(ii)==0
        labels(ii,:)=[1 0 0];
    elseif temp_labels(ii)==1
        labels(ii,:)=[0 1 0];
    else
        labels(ii,:)=[0 0 1];
    end
end
%抽取出特征列
input = matdata.dataset(:,3:6);

%划分训练集和测试集：75%为训练集，25%为测试集
[rows,cols]=size(input);
train_input=input(1:ceil(rows*0.75),:);
train_label=labels(1:ceil(rows*0.75),:);
test_input=input(ceil(rows*0.75)+1:rows , :);
test_label=labels(ceil(rows*0.75)+1:rows,:);


%将特征标准化处理
[train_input, minI, maxI] = premnmx(train_input');
%train_input=train_input';
test_input = tramnmx( test_input' , minI, maxI ) ;
%test_input = test_input';

%创建神经网络
net = newff( minmax(train_input) , [3,3] , { 'logsig' 'logsig' } , 'traingdx' ) ; 

%设置训练参数
net.trainparam.show = 50 ;
net.trainparam.epochs = 500 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.005 ;

%开始训练
net = train( net, train_input , train_label' ) ;


%仿真
Y = sim( net , test_input ); 
%统计识别正确率
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

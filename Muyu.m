%% Step1:Import segmented Data
clear;clc;close all;
cd /Users/IZAYA/Desktop;
data=csvread('CI_bank1.csv',1,4);
%% Step2:Import macro data
macro=csvread('Historic_Domestic.csv',92,2); % from 1998.Q4-->changes in 1999.Q1-->lag for 4Q--->influence on 2000.Q1--->This is our first PD's obeservation period

% transfer 3 indexes to changes
macreturn=(macro(2:end,13:15)./macro(1:end-1,13:15)-1);
macro=[macro(2:end,1:12),macreturn,macro(2:end,16)];

%%%%%%%%%%%% Lag macro variables for 0 1 2 3 4 quarter %%%%%%%%%%%%
lagmac=[];
for i =1:16
    xlag=lagmatrix(macro(:,i),[0 1 2 3 4]);
    lagmac=[lagmac,xlag];
end

% lagmac(isnan(lagmac)) = 0;
% lagmac(isinf(lagmac)) = 0;
lagmac=lagmac(5:end-2,:);
lagmac=lagmac(5:end-3,:);
%This is the lagmacro data from 2000.Q1-2017.Q2 after transfering 3 indexes data

[a,b]=size(lagmac);
%% Step3:Regression
defr=[];
B=[];
Stats=[];
paratick=[];
defaultrate=[];
%%%%%%%%%%%%%%%%%%%%%%%%%% For sicbin i %%%%%%%%%%%%%%%%%%%%%%%%%%
%for sicbin j=1:7, calculate PDs for the 72 period from 2000.Q1 to
%2017.Q4, denoting as 'defr'
for j = 1:7
    for i =j:7:436+j
        defrate=data(i,3);
        defr=[defr;defrate];
    end
    %%%%%%%%%%%%%%%%%%%%% Step3: Simple Linear Regression %%%%%%%%%%%%%%%%
    % Perform SLR for sicbin i for all 16 domestic variable with lags
    T=[];
    for z =1:80
        x=[ones(a,1),lagmac(:,z)];
        [beta,bint,r,rint,stats] = regress(defr,x);
        %stats: 1)the R2 statistic,2)the F statistic,3)its p value,4)estimate of the error variance
        Rsq=stats(:,1);
        pvalue=stats(:,3);
        Ts=[beta(2),pvalue,Rsq];
        %Save [coef,p-value,Rsq] for each SLR in 'T'
        T=[T;Ts];
    end
    T=[(1:80)',T];
    %%%%%%%%%%%%%%%%%%%%%%%% Step4: Select Variables %%%%%%%%%%%%%%%%%%%%%
    % Criterion 1: Choose pvalue< 0.05
    C=[];
    for n = 1:80
        if T(n,3)<0.05
            C(n,:)=T(n,:);
        else
            C(n,:)=[T(n,1),0,0,0];
        end
    end
    % Criterion 2: Then choose variables with top3 largest R-sq
    choose = sortrows(C,4);
    paratick=[paratick;choose(end-2:end,:)];
    %%%%%%%%%%%%%%%%%%%%%%%%% End sicbin j %%%%%%%%%%%%%%%%%%%%%%%%%
    defaultrate=[defaultrate,defr];
    defr=[];
    %clear vector of PD for sicbin j
end
%%Step5: Check for multicollinearity in the selected model
%(This step is conducted in Mintab)
%% Forecasts for each scenario.
% The Coefficents of the selected models are all stored in 'macro_b'
% We need to import the scenario data and transform relevant macro variables
% Note: to prepare the data for importing, I comboined the historical Fed
% data with the scenario datasets from the Fed.  I also replaced the first
% two columns with year and quarters, respectively.
scen{1} = csvread('baseline.csv',1,0);
scen{2} = csvread('adverse.csv',1,0);
scen{3} = csvread('severelyadverse.csv',1,0); %16 variables, 18 columns

for i = 1:3
    date = scen{i}(:,1) + scen{i}(:,2)/4  - 0.25;
    scen{i} = [date,scen{i}];
    [M,N] = size(scen{i});
end %19 colums, the 1st column is

% Convert the indexes to returns:
for i = 1:3
    indexes = scen{i}(:,16:18);
    idx_ret = [nan(1,3);
        (indexes(2:end,:) - indexes(1:end-1,:))./indexes(1:end-1,:)];
    macros = [scen{i}(:,4:15),idx_ret,scen{i}(:,19)]; %16 colums=[12+3+1];
    scen{i}(:,4:19) = macros; %give new macro to senarios{i}, scen=1*19
    % Now generate the additional macro variables we want to test.
    % For this, we are going to use lags of 1 to 4 quarters.
    cols = [4:19];
    
    l1_macro = [nan(1,length(cols));
        scen{i}(1:end - 1,cols)];
    l2_macro = [nan(2,length(cols));
        scen{i}(1:end - 2,cols)];
    l3_macro = [nan(3,length(cols));
        scen{i}(1:end - 3,cols)];
    l4_macro = [nan(4,length(cols));
        scen{i}(1:end - 4,cols)];
    
    scen_full{i} = [scen{i},l1_macro,l2_macro,l3_macro,l4_macro]; %lag[0 1 2 3 4]=19+4*16
    % Retain only the observations from 2000.Q1 to 2021.Q1
    scen_full{i}(scen_full{i}(:,1) < 2000,:) = [];
    % Retain only the macro variables
    scen_full{i}=scen_full{i}(:,4:end);
end

% plot the forecasts of PD over time for our 7 sicbin
time=date(97:159);
b0hat=[.011197 ;-0.001742 ;-0.003205 ;.0000470;.0001404;-0.0027711;-0.03621 ];
b1hat=[0 0 -0.0020569;
    0 0.0006847  -0.02985 ;
    0 0.0011482 -0.023190 ;
    0 0 0.0008011;
    0 0.00011404  -0.007125;
    0.0005172  0 0.020172;
    0 0 0.006823 ];
% plot the historical default rate from 2000.Q1-2017.Q2
figure;
for i =1:7
    subplot(3,3,i);
    x=time;
    y=defaultrate(:,i);
    plot(x,y,'black');
    xlabel('Time')
    ylabel('Quarterly Default Rate')
    mytitle = strcat('Default Rates for SIC Bin~ ',num2str(i));
    title(mytitle);
    hold on;
end

% Plot the forecasts of PD over time for our 7 sicbin
time_fcst=date(97:end);
scen_maro=[];
yhat=[];
for i =1:3
    for z=1:16
        for j =z:16:64+z
            %senario i
            lag_scen=scen_full{i}(:,j);
            scen_maro=[scen_maro,lag_scen]; %85*240
        end
    end
end


hold on;
% baseline
% 2000.Q1-2021.Q1, a 85 quarters period for baseline
yhat(:,1)=b0hat(1)+[zeros(85,1),zeros(85,1),scen_maro(:,41)]*b1hat(1,:)';
yhat(:,2)=b0hat(2)+[zeros(85,1),scen_maro(:,50),scen_maro(:,73)]*b1hat(2,:)';
yhat(:,3)=b0hat(3)+[zeros(85,1),scen_maro(:,55),scen_maro(:,62)]*b1hat(3,:)';
yhat(:,4)=b0hat(4)+[zeros(85,1),zeros(85,1),scen_maro(:,35)]*b1hat(4,:)';
yhat(:,5)=b0hat(5)+[zeros(85,1),scen_maro(:,11),scen_maro(:,63)]*b1hat(5,:)';
yhat(:,6)=b0hat(6)+[scen_maro(:,49),zeros(85,1),scen_maro(:,75)]*b1hat(6,:)';
yhat(:,7)=b0hat(7)+[zeros(85,1),zeros(85,1),scen_maro(:,24)]*b1hat(7,:)';
for  m=1:7
    subplot(3,3,m)
    plot(time_fcst,yhat(:,m),'g')
    
    hold on;
end
hold on;
% adverse
% 2000.Q1-2021.Q1, a 85 quarters period for adverse
yhat2(:,1)=b0hat(1)+[zeros(85,1),zeros(85,1),scen_maro(:,41+80)]*b1hat(1,:)';
yhat2(:,2)=b0hat(2)+[zeros(85,1),scen_maro(:,50+80),scen_maro(:,73+80)]*b1hat(2,:)';
yhat2(:,3)=b0hat(3)+[zeros(85,1),scen_maro(:,55+80),scen_maro(:,62+80)]*b1hat(3,:)';
yhat2(:,4)=b0hat(4)+[zeros(85,1),zeros(85,1),scen_maro(:,35+80)]*b1hat(4,:)';
yhat2(:,5)=b0hat(5)+[zeros(85,1),scen_maro(:,11+80),scen_maro(:,63+80)]*b1hat(5,:)';
yhat2(:,6)=b0hat(6)+[scen_maro(:,49+80),zeros(85,1),scen_maro(:,75+80)]*b1hat(6,:)';
yhat2(:,7)=b0hat(7)+[zeros(85,1),zeros(85,1),scen_maro(:,24+80)]*b1hat(7,:)';
for  m=1:7
    subplot(3,3,m)
    plot(time_fcst,yhat2(:,m),'b')
    
    hold on;
end
% severly adverse
% 2000.Q1-2021.Q1, a 85 quarters period for severly adverse
yhat3(:,1)=b0hat(1)+[zeros(85,1),zeros(85,1),scen_maro(:,41+160)]*b1hat(1,:)';
yhat3(:,2)=b0hat(2)+[zeros(85,1),scen_maro(:,50+160),scen_maro(:,73+160)]*b1hat(2,:)';
yhat3(:,3)=b0hat(3)+[zeros(85,1),scen_maro(:,55+160),scen_maro(:,62+160)]*b1hat(3,:)';
yhat3(:,4)=b0hat(4)+[zeros(85,1),zeros(85,1),scen_maro(:,35+160)]*b1hat(4,:)';
yhat3(:,5)=b0hat(5)+[zeros(85,1),scen_maro(:,11+160),scen_maro(:,63+160)]*b1hat(5,:)';
yhat3(:,6)=b0hat(6)+[scen_maro(:,49+160),zeros(85,1),scen_maro(:,75+160)]*b1hat(6,:)';
yhat3(:,7)=b0hat(7)+[zeros(85,1),zeros(85,1),scen_maro(:,24+160)]*b1hat(7,:)';
for  m=1:7
    subplot(3,3,m)
    plot(time_fcst,yhat3(:,m),'r')
    legendInfo = {'Actual','Baseline','Adverse','Severely Adverse'};
    legend(legendInfo);
    hold on;
end

%% calculate the expected loss for each period
% get the average exposure and its standard deviation from historical data
rec=0.7; % define recover rate = 0.7

%%%%%%%%%%%%%%%%%%%%% Historical Loss %%%%%%%%%%%%%%%%%%%%%
for j =1:7 %for industry j
    expos_sic(:,j)=data(j:7:436+j,2); %historical exposure (pvloan_avg) per industry
    hist_loss(:,j)=expos_sic(:,j).*defaultrate(:,j)*(1-rec);
    %simulate loss for 2017.Q3-2021.Q1 since we only have PD of 2017.Q2
    [avgexp,stdexp]=normfit(expos_sic(:,j));
    for i =1:15
        R(i,j)=abs(normrnd(avgexp,stdexp)); % each column representing each sicbin's future exposures from 2017.Q3-2021.Q1, (15*1)
    end
end

%%%%%%%%%%%%%%%%%%%%% Future Loss %%%%%%%%%%%%%%%%%%%%%
% assume each scen has same exposure,i.e., same R
% yhat-->scen1's PD from 2017.Q3-2021.Q1,(15 quarter * 7 sicbin)
% R-->scen1's exposure from 2017.Q3-2021.Q1,(15 quarter * 7 sicbin)
% 1-rec-->realized loss fraction
loss{1}=yhat(71:end,:)*(1-rec).*R; % loss of scenario1 from 2017.Q3-2021.Q1
loss{2}=yhat2(71:end,:)*(1-rec).*R;% loss of scenario2 from 2017.Q3-2021.Q1
loss{3}=yhat3(71:end,:)*(1-rec).*R;% loss of scenario3 from 2017.Q3-2021.Q1


%% Plot loss from 2018.Q3-2021.Q1
figure;
losspath_sic=[];
time_fcst=time_fcst(1:63+15);
for j=1:3
    for i=1:7
        subplot(3,3,i);
        losspath_sic(:,i)=[hist_loss(:,i);loss{j}(:,i)];
        plot(time_fcst,losspath_sic(:,i));      
        legendInfo = {'Baseline','Adverse','Severely Adverse'};
        legend(legendInfo);   hold on;
        mytitle = strcat('Loss for SIC Bin~ ',num2str(i));
        title(mytitle);
        
    end
    
end

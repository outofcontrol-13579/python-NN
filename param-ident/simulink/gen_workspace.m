clear variables;

meas_folder = '..\..\dime12\datasets\'
filenames = {'Id.csv', 'Iq.csv', 'Wel.csv', 'Ud.csv', 'Uq.csv'}; 
vars = {'id', 'iq', 'om', 'ud', 'uq'}; 

Ts = 1e-5

time = (0:Ts:910415*Ts)';

for i = 1:length(filenames)
    currentFile = append(meas_folder, filenames{i}); 
    currentVar = vars{i};
    temp.(currentVar) = table2array(readtable(currentFile));
    data.(currentVar) = timeseries(temp.(currentVar), time)
end


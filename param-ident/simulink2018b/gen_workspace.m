clear variables;
meas_folder = '..\..\dime12\datasets\';
filenames = {'Id.csv', 'Iq.csv', 'Wel.csv', 'Ud.csv', 'Uq.csv'};
vars = {'id', 'iq', 'om', 'ud', 'uq'};
Ts = 1e-5;
time = (0:Ts:910415*Ts)';

fprintf('Time vector length: %d\n', length(time));

for i = 1:length(filenames)
    currentFile = [meas_folder, filenames{i}];
    currentVar = vars{i};

    rawData = dlmread(currentFile, ',', 0, 0);
    temp.(currentVar) = rawData(:);

    fprintf('File: %s  ->  length: %d\n', filenames{i}, length(temp.(currentVar)));

    data.(currentVar) = timeseries(temp.(currentVar), time);
end
function generateReport(anomalyLog, gazeHeatmap)
    if isempty(anomalyLog)
        fprintf('Exam session completed. Zero flags raised.\n');
        return;
    end
    
    flagTable = struct2table(anomalyLog);
    
    % --- Output 1: Anomaly Timeline Plot ---
    figure('Name', 'Exam Session Anomaly Timeline', 'NumberTitle', 'off');
    typeNames = cellstr(flagTable.Type);
    categoricalTypes = categorical(typeNames);
    
    scatter(flagTable.Time, categoricalTypes, 60, 'filled', 'MarkerFaceColor', [0.85 0.33 0.1]);
    xlabel('Exam Elapsed Time (Seconds)');
    ylabel('Flagged Violation Type');
    title('Proctoring Anomaly Timeline');
    grid on;
    
    % --- Output 2: Smoothed Gaze Heatmap ---
    figure('Name', 'Student Attention Density Heatmap', 'NumberTitle', 'off');
    smoothedHeatmap = imgaussfilt(gazeHeatmap, 25); % Apply spatial blur filter
    
    imagesc(smoothedHeatmap);
    colormap('hot');
    colorbar;
    title('Visual Attention Density Map');
    xlabel('Horizontal Screen Space (Pixels)');
    ylabel('Vertical Screen Space (Pixels)');
    axis image;
    
    % --- Output 3: Summary Text ---
    fprintf('\n=======================================\n');
    fprintf('        EXAM PROCTORING REPORT         \n');
    fprintf('=======================================\n');
    fprintf('Total Anomalies Flagged: %d\n', height(flagTable));
    
    typeNames = cellstr(flagTable.Type);
    types = unique(typeNames);
    for i = 1:length(types)
        typeName = char(types{i});
        count = sum(strcmp(typeNames, typeName));
        fprintf('- %s: %d instances\n', typeName, count);
    end
    fprintf('=======================================\n');
end
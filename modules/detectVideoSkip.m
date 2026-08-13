function [skipFlags, skipState] = detectVideoSkip(currentGray, prevGray, currentTime, skipState)
% DETECTVIDEOSKIP Identifies if a video feed has intentionally lagged/frozen 
% and subsequently jumped to a new frame.

% Initialize empty flag array
skipFlags = struct('Time', {}, 'Type', {}, 'Severity', {});

% If it's the first frame, we have no previous frame to compare against
if isempty(prevGray)
    return;
end

% 1. Similarity Metric (Manual MSE for absolute data-type safety)
% Convert to double to prevent uint8 saturation (where negatives become 0)
currentD = double(currentGray);
prevD = double(prevGray);
mseValue = mean((currentD(:) - prevD(:)).^2);

% 2. Calibrated Thresholds for Compressed Video
T_freeze   = 20.0; 
T_jump     = 150.0; 
T_duration = 2.0;   

% 3. State Machine Logic
if mseValue < T_freeze
    % State 2: Freeze Detected (Tolerates compression noise)
    if ~skipState.isFrozen
        skipState.isFrozen = true;
        skipState.freezeStartTime = currentTime;
    end
else
    % Movement detected! Check if we were previously in a freeze state
    if skipState.isFrozen
        freezeDuration = currentTime - skipState.freezeStartTime;

        % State 3: Did it unfreeze with a massive jump?
        if mseValue > T_jump
            if freezeDuration >= T_duration
                % The jump happened after a sustained freeze - Trigger Anomaly!
                skipFlags(end+1) = struct(...
                    'Time', currentTime, ...
                    'Type', sprintf('Video Skip Detected (Started at %.1fs, Frozen for %.1fs)', skipState.freezeStartTime, freezeDuration), ...
                    'Severity', 'High');
            end
        end

        % Reset the state machine back to State 1 (Active Monitoring)
        skipState.isFrozen = false;
        skipState.freezeStartTime = NaN;
    end
end
end
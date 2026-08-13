function state = initFaceLightState()
%INITFACELIGHTSTATE Create the state struct for detectFaceLightChange.
%
% Call this once before the main processing loop, then pass the returned
% struct into detectFaceLightChange each call and keep reassigning it
% (same pattern as prevIntensity / initialYaw etc. elsewhere in
% main_proctor.m).

    state.windowSize          = 10;    % samples in the rolling baseline
    state.buffer              = nan(1, state.windowSize);
    state.bufIdx              = 0;
    state.lastFlagTime        = -inf;  % seconds
    state.zScoreThreshold     = 1.3;   % std-devs from baseline to flag
    state.absDeltaThreshold   = 8;    % min raw brightness jump (0-255)
    state.minSecBetweenFlags  = 0.5;   % debounce, in seconds not frames
                                        % (frames are subsampled by detectEvery)
end

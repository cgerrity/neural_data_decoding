function root = ndd_add_matlab_paths()
% ndd_add_matlab_paths  Resolve the MATLAB pipeline source tree and add its
% subdirectories to MATLAB's path.
%
% Resolution order (first hit wins):
%   1. NDD_MATLAB_SOURCE_ROOT environment variable.
%   2. <scripts_dir>/../.. — legacy parent-of-project layout
%      (project nested inside the MATLAB repo).
%   3. /Users/cgerrity/Documents/MATLAB/Neural Data Reading — known
%      absolute fallback for the dev workstation.
%
% The chosen root must contain `Processing_Functions_cgg/` or the function
% errors clearly with NDD:MatlabSourceNotFound. All canonical pipeline
% subdirectories that exist under the root are added to the path via
% addpath(genpath(...)). Used by every MATLAB fixture-generator script
% under scripts/.
%
% Returns the resolved root as a char array, so the caller can log it.

    thisDir = fileparts(mfilename('fullpath'));   % .../neural_data_decoding/scripts
    projDir = fullfile(thisDir, '..');            % .../neural_data_decoding

    candidates = {};
    envVal = getenv('NDD_MATLAB_SOURCE_ROOT');
    if ~isempty(envVal)
        candidates{end+1} = envVal;
    end
    candidates{end+1} = fullfile(projDir, '..');                          % legacy nested layout
    candidates{end+1} = '/Users/cgerrity/Documents/MATLAB/Neural Data Reading';  % known fallback

    root = '';
    tried = {};
    for k = 1:numel(candidates)
        c = candidates{k};
        tried{end+1} = c; %#ok<AGROW>
        if isfolder(fullfile(c, 'Processing_Functions_cgg'))
            root = c;
            break;
        end
    end

    if isempty(root)
        msg = sprintf(['MATLAB source tree not found. Looked for ' ...
            '''Processing_Functions_cgg/'' under:\n']);
        for k = 1:numel(tried)
            msg = [msg sprintf('  - %s\n', tried{k})]; %#ok<AGROW>
        end
        msg = [msg sprintf(['\nSet NDD_MATLAB_SOURCE_ROOT to the directory ' ...
            'containing ''Processing_Functions_cgg/'' to override.'])];
        error('NDD:MatlabSourceNotFound', '%s', msg);
    end

    for sub = {'Processing_Functions_cgg', 'FLU_Process_scripts_LT', ...
               'LoopUtil', 'exp-utils-cjt-4', 'External_Functions', ...
               'YAMLMatlab_0.4.3'}
        p = fullfile(root, sub{1});
        if isfolder(p)
            addpath(genpath(p));
        end
    end
end

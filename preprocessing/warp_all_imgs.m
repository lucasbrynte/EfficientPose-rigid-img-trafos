DRY_RUN = false;

% Path to LINEMOD datasets (as provided by EfficientPose):
data_path = '/path/to/Linemod_preprocessed';

% LINEMOD intrinsic camera parameters
K = [
  572.4114         0  325.2611;
         0  573.5704  242.0490;
         0         0    1.0000
];

one_based_indexing = false;

obj_ids = {'01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15'};
subdirs = {'rgb', 'mask', 'merged_masks'};

for j = 1:length(obj_ids)
    obj_id = obj_ids{j};
    for k = 1:length(subdirs)
        subdir = subdirs{k};
        if strcmp(subdir, 'rgb')
            interpolation_method = 'bilinear';
        else
            interpolation_method = 'nearest_neighbor';
        end
        in_dir = [data_path '/data/' obj_id filesep subdir];
        if ~exist(in_dir, 'dir')
            continue
        end
        out_dir = [data_path '/data/' obj_id filesep subdir '_arctan_warped'];
        if ~DRY_RUN
            [~, ~, ~] = mkdir(out_dir);
        end
        in_paths = dir([in_dir filesep '*.png']);
        for l = 1:length(in_paths)
            in_dir = in_paths(l).folder;
            fname = in_paths(l).name;
            in_path = [in_dir filesep fname];
            fprintf('%s\n', in_path);
            out_path = [out_dir filesep fname];
            I = imread(in_path);
            I_warped = warp_an_image(I, K, interpolation_method, one_based_indexing);
            if ~DRY_RUN
                imwrite(I_warped, out_path);
            end
        end
    end
end

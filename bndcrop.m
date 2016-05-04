function u = bndcrop(v, sz)
    u = zeros(sz, class(v));
    u(1:size(u,1), 1:size(u,2), 1:size(u,3), :) = v(1:size(u,1), 1:size(u,2), 1:size(u,3), :);
return
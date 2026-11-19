function [img_size, vs] = R2_GetDIRLabParams(case_id)
    if case_id <= 5
        img_size = [256, 256, 94]; vs = [0.97, 0.97, 2.5];
    else
        img_size = [512, 512, 128]; vs = [0.97, 0.97, 2.5];
    end
    switch case_id
        case 1, img_size(3) = 94;
        case 2, img_size(3) = 112; vs(1:2) = [1.16, 1.16];
        case 3, img_size(3) = 104; vs(1:2) = [1.15, 1.15];
        case 4, img_size(3) = 99;  vs(1:2) = [1.13, 1.13];
        case 5, img_size(3) = 106; vs(1:2) = [1.10, 1.10];
        case 7, img_size(3) = 136;
        case 10, img_size(3) = 120;
    end
end
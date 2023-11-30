# -*- coding = utf-8 -*-
# import cv2
import torch
import numpy as np

def add_trigger(args, image, test=False):
    # pixel_max = max(1,torch.max(image))

    if args.trigger == 'square':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1

        if args.dataset == 'cifar':
            pixel_max = 1
        image[:, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = pixel_max
    elif args.trigger == 'pattern':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, args.triggerY + 0, args.triggerX + 0] = pixel_max
        image[:, args.triggerY + 1, args.triggerX + 1] = pixel_max
        image[:, args.triggerY - 1, args.triggerX + 1] = pixel_max
        image[:, args.triggerY + 1, args.triggerX - 1] = pixel_max

    return image
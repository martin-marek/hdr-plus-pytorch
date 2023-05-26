import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functorch.dim import dims

from torch import Tensor
from typing import List


@torch.jit.script
def upscale_previous_alignment(alignment: Tensor,
                               downscale_factor: int,
                               n_tiles_x: int, n_tiles_y: int
                              ) -> Tensor:
    """
    When layers in an image pyramid are iteratively compared,
    the absolute pixel distances in each layer represent different
    relative distances. This function interpolates an optical flow
    from one resolution to another, taking care to scale the values.
    """
    alignment = alignment[None].float() # [1, 2, n_tiles_y, n_tiles_x]
    alignment = F.interpolate(alignment, size=(n_tiles_y, n_tiles_x), mode='nearest')
    alignment *= downscale_factor # [1, 2, n_tiles_y, n_tiles_x]
    alignment = (alignment[0]).to(torch.int16) # [2, n_tiles_y, n_tiles_x]
    return alignment


@torch.jit.script
def build_pyramid(image: Tensor,
                  downscale_factor_list: List[int],
                 ) -> List[Tensor]:
    """
    Create an image pyramid from a single image.
    """
    # if the input image has multiple channels (e.g. RGB), average them to obtain a single-channel image
    layer = torch.mean(image, 0, keepdim=True)

    # iteratively build each level in the image pyramid
    pyramid = []
    for downscale_factor in downscale_factor_list:
        layer = F.avg_pool2d(layer, downscale_factor)
        pyramid.append(layer)
    return pyramid


def align_layers(ref_layer: Tensor,
                 comp_layer: Tensor,
                 prev_alignment: Tensor,
                 tile_size: int,
                 search_dist: int,
                 downscale_factor: int = 1
                ) -> Tensor:
    """
    Estimates the optical flow between layers of two distinct image pyramids.
    
    Args:
        comp_layer: the layer to be aligned to `ref_layer`
        prev_alignment: alignment from a coarser pyramid layer
        downscale_factor: scaling factor between the previous layer and current layer, only required if `prev_alignment` is not zeros
    """
    device = ref_layer.device

    # compute number of tiles in a layer such that they overlap
    n_channels, layer_height, layer_width = ref_layer.shape
    n_tiles_y = layer_height // (tile_size // 2) - 1
    n_tiles_x = layer_width  // (tile_size // 2) - 1
    
    # upscale previous alignment
    prev_alignment = upscale_previous_alignment(prev_alignment, downscale_factor, n_tiles_x, n_tiles_y)

    # get reference image tiles (no shift)
    channel, tile_idx_y, tile_idx_x, tile_h, tile_w = dims(sizes=[None, n_tiles_y, n_tiles_x, tile_size, tile_size])
    x_min = torch.linspace(0, layer_width-tile_size,  n_tiles_x, dtype=torch.int16, device=device)[tile_idx_x]
    y_min = torch.linspace(0, layer_height-tile_size, n_tiles_y, dtype=torch.int16, device=device)[tile_idx_y]
    x = x_min + tile_w
    y = y_min + tile_h
    ref_tiles = ref_layer[channel, y, x]

    # get comparison image tiles (shifted)
    shift_x, shift_y = dims(sizes=[1+2*search_dist, 1+2*search_dist])
    x = (x + prev_alignment[0, tile_idx_y, tile_idx_x] + (shift_x - search_dist)).clip(0, layer_width-1)
    y = (y + prev_alignment[1, tile_idx_y, tile_idx_x] + (shift_y - search_dist)).clip(0, layer_height-1)
    comp_tiles = comp_layer[channel, y, x]

    # compute the difference between the reference and comparison tiles
    diff = (ref_tiles - comp_tiles).abs().sum([channel, tile_w, tile_h])
    diff = diff.order(tile_idx_y, tile_idx_x, (shift_y, shift_x))

    # set the difference value for tiles outside of the frame to infinity
    tile_is_inside_layer = ((x<0)^(x>=layer_width)).sum(tile_w) + ((y<0)^(y>=layer_height)).sum(tile_h) == 0
    tile_is_inside_layer = tile_is_inside_layer.order(tile_idx_y, tile_idx_x, (shift_y, shift_x))
    diff[~tile_is_inside_layer] = float('inf')

    # find which shift (dx, dy) between the reference and comparison tiles yields the lowest loss
    min_idx = torch.argmin(diff, -1)
    dy = min_idx // (2*search_dist+1) - search_dist
    dx = min_idx %  (2*search_dist+1) - search_dist

    # save the current alignment
    alignment = torch.stack([dx, dy], 0) # [2, n_tiles_y, n_tiles_x]

    # combine the current alignment with the previous alignment
    alignment += prev_alignment

    return alignment


@torch.jit.script
def warp_images(images: Tensor,
                alignments: Tensor
               ) -> Tensor:
    """
    Given a batch of images and their optical
    flows with respect to a reference image,
    warps the images using the given optical flow.
    """
    device = images.device
    # start off with a generic tensor of indices of each pixel
    N, C, H, W = images.shape
    grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([grid[1], grid[0]], dim=0).float()
    grid = grid[None].repeat_interleave(repeats=N, dim=0)

    # apply the optical flow to the indices
    grid += alignments
    
    # scale the indices to [-1, 1] in x and y
    grid[:, 0] = (grid[:, 0] / (W-1))*2 - 1
    grid[:, 1] = (grid[:, 1] / (H-1))*2 - 1
    
    # apply the grid of indices to warp the images
    grid = grid.permute(0, 2, 3, 1)
    images_warped = F.grid_sample(images, grid, mode='nearest', align_corners=True, padding_mode='zeros')
    
    return images_warped


def align_and_merge(images: Tensor,
                    ref_idx: int = 0,
                    device: torch.device = torch.device('cpu'),
                    min_layer_res: int = 64,
                    tile_size: int = 16,
                    search_dist: int = 2,
                   ) -> Tensor:
    """
    Align and merge a burst of images. The input and output tensors are assumed to be on CPU device, to reduce GPU memory requirements.

    Args:
        images: burst of shape (num_frames, channels, height, width)
        ref_idx: index of the reference image (all images are alinged to this image)
        device: the PyTorch device to use (either 'cpu' or 'cuda')
        min_layer_res: size of the smallest pyramid layer
        tile_size: size of tiles in each pyramid layer
    """
        
    # check the shape of the burst
    N, C, H, W = images.shape
    
    # compute the necessary number of layers in the pyramid assuming
    # that each layer has half the resolution of the previous layer
    res = min(H, W)
    n_layers = 0
    while res > min_layer_res:
        n_layers += 1
        res /= 2
    
    # build a pyramid from the reference image
    downscale_factor_list = n_layers*[2]
    ref_idx = torch.tensor(ref_idx)
    ref_image = images[ref_idx].to(device)
    ref_pyramid = build_pyramid(ref_image, downscale_factor_list)
    
    # iterate through the comparison images
    merged_image = ref_image.clone() / N
    comp_idxs = torch.arange(N)[torch.arange(N)!=ref_idx]
    for i, comp_idx in enumerate(comp_idxs):

        # build a pyramid from the comparison image
        comp_image = images[comp_idx].to(device)
        comp_pyramid = build_pyramid(comp_image, downscale_factor_list)

        # start off with default alignment (no shift between images)
        alignment = torch.zeros([2, 1, 1], device=device)
        
        # iteratively improve the alignment in each pyramid layer
        for layer_idx in torch.flip(torch.arange(len(ref_pyramid)), [0]):
            downscale_factor = downscale_factor_list[min(layer_idx+1, len(ref_pyramid)-1)]
            alignment = align_layers(ref_pyramid[layer_idx], comp_pyramid[layer_idx],
                                     alignment, tile_size, search_dist, downscale_factor)
            
        # scale the alignment to the resolution of the original image
        alignment = upscale_previous_alignment(alignment, downscale_factor_list[0], W, H)
        
        # warp the comparison image based on the computed alignment
        comp_image_aligned = warp_images(comp_image[None], alignment[None])[0]

        # add the aligned image to the output
        merged_image += comp_image_aligned / N

    merged_image = merged_image.cpu()
    
    return merged_image

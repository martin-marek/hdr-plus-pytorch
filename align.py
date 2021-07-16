import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch import Tensor
from typing import Tuple, List, Optional


@torch.jit.script
def n_tiles(layer_h: int, layer_w: int,
            tile_h: int, tile_w: int
           ) -> Tuple[int, int]:
    """
    Compute the desired number of tiles in a layer,
    such that they overlap.
    """
    n_tiles_y = layer_h // (tile_h // 2) - 1
    n_tiles_x = layer_w // (tile_w // 2) - 1
    return n_tiles_x, n_tiles_y


@torch.jit.script
def tile_indices(layer_w: int, layer_h: int,
                 tile_w: int, tile_h: int,
                 n_tiles_x: int, n_tiles_y: int,
                 device: torch.device
                ) -> Tuple[Tensor, Tensor]:
    """
    Computes the indices of tiles along a layer.
    """
    # compute x indices
    x_min = torch.linspace(0, layer_w-tile_w-1, n_tiles_x, dtype=torch.int16, device=device) # [n_tiles_x]
    dx = torch.arange(0, tile_w, device=device) # [tile_w]
    x = x_min[:, None] + dx[None, :] # [n_tiles_x, tile_w]
    x = x[None, :, None, :].repeat(n_tiles_y, 1, tile_h, 1) # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    
    # compute y indices
    y_min = torch.linspace(0, layer_h-tile_h-1, n_tiles_y, dtype=torch.int16, device=device) # [n_tiles_y]
    dy = torch.arange(0, tile_h, device=device) # [tile_h]
    y = y_min[:, None] + dy[None, :] # [n_tiles_y, tile_h]
    y = y[:, None, :, None].repeat(1, n_tiles_x, 1, tile_w) # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    
    return x, y # (2 x [n_tiles_y, n_tiles_x, tile_h, tile_w])


@torch.jit.script
def shift_indices(x: Tensor, y: Tensor,
                  search_dist_min: int,
                  search_dist_max: int
                 ) -> Tuple[Tensor, Tensor]:
    """
    Given a tensor of tile indices, shift these indices by
    [search_dist_min, search_dist_max] in both x- and y-directions.
    This creates (search_dist_max - search_dist_min + 1)^2
    tile "displacements". All the displacements are returned
    along the first (batch) dimension. To recover the displacement
    from the batch dimension, use the following conversion:
    >>> dy = idx // n_pos + search_dist_min
    >>> dx = idx % n_pos + search_dist_min
    """
    # create a tensor of displacements
    device = x.device
    ds = torch.arange(search_dist_min, search_dist_max+1, device=device)
    n_pos = len(ds)
    
    # compute x indices
    x = x[:, :, :, :, None, None] + ds[None, None, None, None, None, :] # [n_tiles_y, n_tiles_x, tile_h, tile_w, 1, n_pos]
    x = x.repeat(1, 1, 1, 1, n_pos, 1) # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos, n_pos]
    x = x.flatten(-2) # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos*n_pos]
    x = x.permute(4, 0, 1, 2, 3) # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    
    # compute y indices
    y = y[:, :, :, :, None, None] + ds[None, None, None, None, :, None] # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos, 1]
    y = y.repeat(1, 1, 1, 1, 1, n_pos) # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos, n_pos]
    y = y.flatten(-2) # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos*n_pos]
    y = y.permute(4, 0, 1, 2, 3) # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    
    return x, y # (2 x [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w])


@torch.jit.script
def clamp(x: Tensor, y: Tensor,
          layer_w: int, layer_h: int
         ) -> Tuple[Tensor, Tensor]:
    """
    Clamp indices to layer shape.
    """
    # use inplace operations to reduce momory usage
    x.clamp_(0, layer_w-1)
    y.clamp_(0, layer_h-1)
    return x, y


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


@torch.jit.script
def align_layers(ref_layer: Tensor,
                 comp_layer: Tensor,
                 tile_shape: List[int],
                 search_region: List[int],
                 prev_alignment: Tensor,
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
    # compute dimensions of layer and tiles
    _, layer_h, layer_w = ref_layer.shape
    tile_w, tile_h = tile_shape
    n_tiles_x, n_tiles_y = n_tiles(layer_h, layer_w, tile_h, tile_w)
    search_dist_min, search_dist_max = search_region
    n_pos = search_dist_max - search_dist_min + 1

    # gather tiles from the reference layer
    x, y = tile_indices(layer_w, layer_h, tile_w, tile_h, n_tiles_x, n_tiles_y, device) # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    ref_tiles = ref_layer[:, y, x] # [1, n_tiles_y, n_tiles_x, tile_h, tile_w]

    # compute coordinates for comparison tiles
    x, y = tile_indices(layer_w, layer_h, tile_w, tile_h, n_tiles_x, n_tiles_y, device) # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    prev_alignment = upscale_previous_alignment(prev_alignment, downscale_factor, n_tiles_x, n_tiles_y) # [2, n_tiles_y, n_tiles_x]
    x += prev_alignment[0, :, :, None, None]
    y += prev_alignment[1, :, :, None, None]
    x, y = shift_indices(x, y, search_dist_min, search_dist_max) # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    
    # check if each comparison tile is fully within the layer dimensions
    tile_is_inside_layer = torch.ones([n_tiles_y, n_tiles_x], dtype=torch.bool, device=device)
    tile_is_inside_layer &= x[:, :, :,  0,  0] >= 0
    tile_is_inside_layer &= x[:, :, :,  0, -1] < layer_w
    tile_is_inside_layer &= y[:, :, :,  0,  0] >= 0
    tile_is_inside_layer &= y[:, :, :, -1,  0] < layer_h
    
    # gather tiles from the comparison layer
    x, y = clamp(x, y, layer_w, layer_h) # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_tiles = comp_layer[0, y, x] # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]

    # compute the difference between the comparison and reference tiles
    diff = comp_tiles - ref_tiles # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    diff = diff.abs().sum(dim=[-2, -1]) # [n_pos*n_pos, n_tiles_y, n_tiles_x]
    
    # set the difference value for tiles outside of the frame to infinity
    diff[~tile_is_inside_layer] = float('inf')

    # find which shift (dx, dy) between the reference and comparison tiles yields the lowest loss
    argmin = diff.argmin(0) # [n_tiles_y, n_tiles_x]
    dy = argmin // n_pos + search_dist_min # [n_tiles_y, n_tiles_x]
    dx = argmin % n_pos + search_dist_min # [n_tiles_y, n_tiles_x]

    # save the current alignment
    alignment = torch.stack([dx, dy], 0) # [2, n_tiles_y, n_tiles_x]
    
    # combine the current alignment with the previous alignment
    alignment += prev_alignment
    
    return alignment # [2, n_tiles_y, n_tiles_x]


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
    grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
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


# @torch.jit.script
def align_and_merge(images: Tensor,
                    ref_idx: int = 0,
                    device: torch.device = torch.device('cpu'),
                    downscale_factor_list: Optional[List[int]] = None,
                    tile_shape_list: Optional[List[List[int]]] = None,
                    search_region_list: Optional[List[List[int]]] = None
                   ) -> Tensor:
    """
    Align and merge a burst of images. The input and output tensors are assumed to be on CPU device, to reduce GPU memory requirements.

    Args:
        images: burst of shape (num_frames, channels, height, width)
        ref_idx: index of the reference image (all images are alinged to this image)
        device: the PyTorch device to use (either 'cpu' or 'cuda')
        downscale_factor_list: scaling factor between image pyramid layers
        tile_shape_list: shape of tiles in each pyramid layer
        search_region_list: search distance [min, max] for each tile in each pyramid layer
    """
    
    # process args
    # - torchscript doesn't support lists as default values, so for some
    #   args, the default is None and the lists are instantiated here
    # - notice that downscale_factor_list[0]==2, i.e. the finest alignment uses
    #   an image downsampled by a factor of 2 – this is to ensure that the RAW images 
    #   are processed in a sensible way (and to speed up the computation)
    if downscale_factor_list is None: downscale_factor_list = [2, 2, 4, 4]
    if tile_shape_list is None: tile_shape_list = [[16, 16], [16, 16], [16, 16], [16, 16]]
    if search_region_list is None: search_region_list = [[-1, 1], [-4, 4], [-4, 4], [-4, 4]]
        
    # check the shape of the burst
    N, C, H, W = images.shape

    # build a pyramid from the reference image
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
        
        # iterative improve the alignment in each pyramid layer
        for layer_idx in torch.flip(torch.arange(len(ref_pyramid)), [0]):
            alignment = align_layers(ref_pyramid[layer_idx],
                             comp_pyramid[layer_idx],
                             tile_shape_list[layer_idx],
                             search_region_list[layer_idx],
                             alignment,
                             downscale_factor_list[min(layer_idx+1, len(ref_pyramid)-1)])
            
        # scale the alignment to the resolution of the original image
        alignment = upscale_previous_alignment(alignment, downscale_factor_list[0], W, H)
        
        # warp the comparison image based on the computed alignment
        comp_image_aligned = warp_images(comp_image[None], alignment[None])[0]

        # add the aligned image to the output
        merged_image += comp_image_aligned / N

    merged_image = merged_image.cpu()
    
    return merged_image

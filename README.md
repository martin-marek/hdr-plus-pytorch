# HDR+ PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/hdr-plus-pytorch/blob/main/demo.ipynb)

This is a simplified PyTorch implementation of HDR+, the backbone of computational photography in Google Pixel phones, described in [Burst photography for high dynamic range and low-light imaging on mobile cameras](http://static.googleusercontent.com/media/www.hdrplusdata.org/en//hdrplus.pdf).

Using an 11GB GPU, alignment works for up to 3MP grayscale images (same as the official implementation), at ~100 ms / image. 
 
# Example

I took a burst of 35 images at ISO 12,800 on Sony RX100-V and boosted it by +2EV. Here's a comparison of a [single image](https://github.com/martin-marek/hdr-plus-pytorch/raw/main/illustrations/burst_sample.jpg) from the burst vs. a [merge of all the images](https://github.com/martin-marek/hdr-plus-pytorch/raw/main/illustrations/merged_image.jpg).
 
![alt text](illustrations/before_and_after.jpg)

# Usage

Here's a minimal example to align and merge a burst. For more, see the [Colab Notebook](https://colab.research.google.com/github/martin-marek/hdr-plus-pytorch/blob/main/demo.ipynb).

```python
import torch
import align

# load image burst
reference_image = torch.zeros([3, 1000, 1000], dtype=torch.float16, device='cuda')
comparison_images = torch.zeros([10, 3, 1000, 1000], dtype=torch.float16, device='cuda')

# align
aligned_images = align.align_images(reference_image, comparison_images)

# merge
merged_image = (reference_image + aligned_images.sum(0)) / (1 + len(aligned_images))
merged_image = torch.clip(merged_image, 0, 1)
```

# Implementation details

The core of my implementation is stacking all tile displacements along the batch dimension and performing comparisons with the help of broadcasting. I've illustrated this for the simplest case of 9 displacements of a 5x5 tile. In reality, the number of tiles and displacements is large. I've annotated the shape of most tensors in my code, so that it's easy to see what's going on in every line.

![alt text](illustrations/tiles.png)

# Features
- [x] jpeg support
- [ ] RAW support
- [x] simple merge
- [ ] robust merge
- [x] tile comparison in pixel space
- [ ] tile comparison in Fourier space
- [x] CUDA support
- [ ] CPU support (requires float32 instead of float16 for some ops)
- [ ] color post-processing
- [ ] automatic selection of the reference image

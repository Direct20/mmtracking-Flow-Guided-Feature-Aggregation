# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    print(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

def flow_warp_feats(x, flow):
    """Use flow to warp feature map.

    Args:
        x (Tensor): of shape (N, C, H_x, W_x).
        flow (Tensor): of shape (N, C, H_f, W_f).

    Returns:
        Tensor: The warpped feature map with shape (N, C, H_x, W_x).
    """
    assert len(x.shape) == 4
    assert len(flow.shape) == 4 and flow.shape[1] == 2
    # 1. resize the resolution of flow to be the same as x.
    size = (x.shape[-2], x.shape[-1])
    flow = resize(
        flow, size=size, mode='bilinear', align_corners=False)
    scale_factor = float(x.shape[-1]) / flow.shape[-1]
    flow = flow * scale_factor

    # 2. compute the flow_field (grid in the code) used to warp features.
    H, W = x.shape[-2:]
    h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W))
    # [1, 1, H, W]
    h_grid = h_grid.to(flow)[None, None, ...]
    # [1, 1, H, W]
    w_grid = w_grid.to(flow)[None, None, ...]
    # [1, 2, H, W]
    grid = torch.cat((w_grid, h_grid), dim=1)
    # [N, 2, H, W]
    grid = grid + flow
    grid[:, 0] = grid[:, 0] / W * 2 - 1
    grid[:, 1] = grid[:, 1] / H * 2 - 1
    # [N, H, W, 2]
    grid = grid.permute(0, 2, 3, 1)

    # 3. warp features.
    x_warp = torch.nn.functional.grid_sample(
        x, grid, padding_mode='border', align_corners=True)
    return x_warp

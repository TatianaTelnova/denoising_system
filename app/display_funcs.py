import matplotlib.pyplot as plt
import torch


def display_img_tensor(img_tensor: torch.Tensor, img_save: bool = False, img_filename: str = None):
    assert img_tensor.shape[0] == 3  # 3 channels (RGB)
    img_plt = torch.tensor(img_tensor).squeeze().permute(1, 2, 0)
    plt.imshow(img_plt)
    if img_save:
        if img_filename is None:
            img_filename = 'img.png'
        plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    plt.show()

# test_dataset.py
from c3vd_dataset import C3VDDataset
from torch.utils.data import DataLoader

dataset = C3VDDataset()
print(f"Số cặp frame: {len(dataset)}")  # 275

batch = dataset[0]
print("color shape:     ", batch['color'].shape)       # (3, 512, 640)
print("color_next shape:", batch['color_next'].shape)  # (3, 512, 640)
print("depth_gt shape:  ", batch['depth_gt'].shape)    # (1, 512, 640)
print("pose_rel shape:  ", batch['pose_rel'].shape)    # (4, 4)
print("depth min/max:   ",
      batch['depth_gt'].nanmean().item())              # ~vài chục mm

loader = DataLoader(dataset, batch_size=4,
                    shuffle=True, num_workers=2)
b = next(iter(loader))
print("Batch color:", b['color'].shape)  # (4, 3, 512, 640)

from gcommand_loader import GCommandLoader
import torch

dataset = GCommandLoader('./gcommands/train')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        # num_workers=4,
        pin_memory=True, sampler=None)

for k, (input_, label) in enumerate(test_loader):
    print(input_.size(), len(label))

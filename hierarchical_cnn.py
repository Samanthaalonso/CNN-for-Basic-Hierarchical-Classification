import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicHierarchicalCNN(nn.Module):
    def __init__(self):
        super(BasicHierarchicalCNN, self).__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers for hierarchical classification
        # Level 1: General category
        self.fc1 = nn.Linear(64 * 56 * 56, 100)
        # Level 2: Subcategory (assuming three branches from level 1)
        self.fc2_1 = nn.Linear(100, 50)
        self.fc2_2 = nn.Linear(100, 50)
        self.fc2_3 = nn.Linear(100, 50)

        # Final classification layers for each branch
        self.fc_final_1 = nn.Linear(50, 10)  # Example: 10 classes in the first branch
        self.fc_final_2 = nn.Linear(50, 8)   # Example: 8 classes in the second branch
        self.fc_final_3 = nn.Linear(50, 6)   # Example: 6 classes in the third branch

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        
        # General category decisions
        x = F.relu(self.fc1(x))
        
        # Branch-specific processing
        x1 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x))
        x3 = F.relu(self.fc2_3(x))

        # Final classification for each branch
        out1 = self.fc_final_1(x1)
        out2 = self.fc_final_2(x2)
        out3 = self.fc_final_3(x3)

        return out1, out2, out3

# Example instantiation and forward pass
model = BasicHierarchicalCNN()
input_tensor = torch.randn(1, 3, 224, 224)  # Simulated batch of one RGB image of size 224x224
outputs = model(input_tensor)
print(outputs)

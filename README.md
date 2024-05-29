# Basic Hierarchical CNN

This repository contains the implementation of a Basic Hierarchical Convolutional Neural Network (CNN) using PyTorch. The model is designed for hierarchical classification tasks, where the classification is performed in a two-level hierarchy.

## Model Architecture

The `BasicHierarchicalCNN` class defines a CNN model with shared convolutional layers followed by branch-specific fully connected layers for hierarchical classification.

### Shared Convolutional Layers

1. `conv1`: A 2D convolutional layer with 32 filters, a kernel size of 5, stride of 1, and padding of 2.
2. `pool`: A max pooling layer with a kernel size of 2 and stride of 2.
3. `conv2`: A 2D convolutional layer with 64 filters, a kernel size of 5, stride of 1, and padding of 2.

### Fully Connected Layers

- **Level 1: General Category**
  - `fc1`: A fully connected layer with 100 units.
  
- **Level 2: Subcategory (assuming three branches from level 1)**
  - `fc2_1`: A fully connected layer with 50 units for the first branch.
  - `fc2_2`: A fully connected layer with 50 units for the second branch.
  - `fc2_3`: A fully connected layer with 50 units for the third branch.

- **Final Classification Layers**
  - `fc_final_1`: A fully connected layer with 10 units for the first branch (e.g., 10 classes).
  - `fc_final_2`: A fully connected layer with 8 units for the second branch (e.g., 8 classes).
  - `fc_final_3`: A fully connected layer with 6 units for the third branch (e.g., 6 classes).

## Forward Pass

The forward pass involves the following steps:

1. Apply `conv1` followed by ReLU activation and `pool`.
2. Apply `conv2` followed by ReLU activation and `pool`.
3. Flatten the output of the pooling layer.
4. Pass through `fc1` with ReLU activation.
5. Branch-specific processing:
   - Pass through `fc2_1` with ReLU activation for branch 1.
   - Pass through `fc2_2` with ReLU activation for branch 2.
   - Pass through `fc2_3` with ReLU activation for branch 3.
6. Final classification:
   - Pass through `fc_final_1` for branch 1 output.
   - Pass through `fc_final_2` for branch 2 output.
   - Pass through `fc_final_3` for branch 3 output.


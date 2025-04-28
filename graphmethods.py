# Standard library imports
import numpy as np
import cv2

# Scientific computing imports
from skimage.util import view_as_blocks

# PyTorch imports
import torch

def calculate_similarity(block1, block2):
    # Convert blocks to grayscale
    block1_gray = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
    block2_gray = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)

    # Sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    block1_sobel = cv2.filter2D(block1_gray, -1, sobel_x) + cv2.filter2D(block1_gray, -1, sobel_y)
    block2_sobel = cv2.filter2D(block2_gray, -1, sobel_x) + cv2.filter2D(block2_gray, -1, sobel_y)

    # Gaussian operator
    gaussian_kernel = cv2.getGaussianKernel(ksize=3, sigma=1)
    block1_gaussian = cv2.filter2D(block1_gray, -1, gaussian_kernel)
    block2_gaussian = cv2.filter2D(block2_gray, -1, gaussian_kernel)

    # Combine Sobel and Gaussian similarities
    sobel_similarity = np.sum(np.abs(block1_sobel - block2_sobel))
    gaussian_similarity = np.sum(np.abs(block1_gaussian - block2_gaussian))

    # Final similarity score
    similarity = sobel_similarity + gaussian_similarity
    return similarity

def build_adjacency_matrix(image, block_size, k):
    # Calculate padding needed to make dimensions divisible by block_size
    height, width = image.shape[0], image.shape[1]
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size
    
    # Pad the image if needed
    if pad_height > 0 or pad_width > 0:
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')

    # ??????blocks
    blocks = view_as_blocks(image, block_shape=(block_size, block_size, 3))
    num_blocks_y, num_blocks_x = blocks.shape[0], blocks.shape[1]
    blocks = blocks.reshape(-1, block_size, block_size, 3)  # ??blocks??

    # ???????
    num_nodes = num_blocks_y * num_blocks_x
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # ????blocks??????
    for i in range(num_nodes):
        for j in range(max(0, i - 2), min(num_nodes, i + 3)):  # ????????
            if i != j:
                sim = calculate_similarity(blocks[i], blocks[j])
                adjacency_matrix[i, j] = sim
                adjacency_matrix[j, i] = sim  # ???

    # ???????k???????
    for i in range(num_nodes):
        neighbors = np.argsort(-adjacency_matrix[i])[:k + 1]  # ?????k????,+1??????
        adjacency_matrix[i] = 0  # ???
        adjacency_matrix[i, neighbors] = 1  # ?????k????1

    return adjacency_matrix


def build_adjacency_matrices(dataloader, block_size, k=5):
    """
    ?????dataloader,???batch?????????????????
    :param dataloader: ??? DataLoader,????????? batchsize*3*640*360
    :param block_size: ?????
    :param k: ???????
    :return: ????????
    """
    all_batches_adj_matrices = []

    for batch_images, _ in dataloader:
        # ????????????
        assert batch_images.shape[2] == 320 and batch_images.shape[3] == 320, "Image dimensions must be 640x360."

        batch_adj_matrices = []
        # ??????
        for image in batch_images:
            # ?????????build_adjacency_matrix??
            image_np = image.permute(1, 2, 0).numpy()  # ???360*640*3
            adj_matrix = build_adjacency_matrix(image_np, block_size, k)
            batch_adj_matrices.append(adj_matrix)

        # ????????????
        full_adj_matrix = np.block([[adj if i == j else np.zeros_like(adj)
                                     for j, adj in enumerate(batch_adj_matrices)]
                                    for i, adj in enumerate(batch_adj_matrices)])
        all_batches_adj_matrices.append(full_adj_matrix)

    return all_batches_adj_matrices
def dense_to_sparse(adj_matrix):
    src, dst = np.where(adj_matrix.cpu().numpy() != 0)
    edge_index_np = np.stack([src, dst])
    edge_index = torch.from_numpy(edge_index_np).long().to(adj_matrix.device)

    return edge_index

# Standard library imports
import numpy as np
import cv2

# Scientific computing imports
from skimage.util import view_as_blocks

# PyTorch imports
import torch

def calculate_similarity(block1, block2):
    # Convert blocks to grayscale
    block1_gray = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
    block2_gray = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)

    # Sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    block1_sobel = cv2.filter2D(block1_gray, -1, sobel_x) + cv2.filter2D(block1_gray, -1, sobel_y)
    block2_sobel = cv2.filter2D(block2_gray, -1, sobel_x) + cv2.filter2D(block2_gray, -1, sobel_y)

    # Gaussian operator
    gaussian_kernel = cv2.getGaussianKernel(ksize=3, sigma=1)
    block1_gaussian = cv2.filter2D(block1_gray, -1, gaussian_kernel)
    block2_gaussian = cv2.filter2D(block2_gray, -1, gaussian_kernel)

    # Combine Sobel and Gaussian similarities
    sobel_similarity = np.sum(np.abs(block1_sobel - block2_sobel))
    gaussian_similarity = np.sum(np.abs(block1_gaussian - block2_gaussian))

    # Final similarity score
    similarity = sobel_similarity + gaussian_similarity
    return similarity

def build_adjacency_matrix(image, block_size, k):
    # Calculate padding needed to make dimensions divisible by block_size
    height, width = image.shape[0], image.shape[1]
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size
    
    # Pad the image if needed
    if pad_height > 0 or pad_width > 0:
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')

    # ??????blocks
    blocks = view_as_blocks(image, block_shape=(block_size, block_size, 3))
    num_blocks_y, num_blocks_x = blocks.shape[0], blocks.shape[1]
    blocks = blocks.reshape(-1, block_size, block_size, 3)  # ??blocks??

    # ???????
    num_nodes = num_blocks_y * num_blocks_x
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # ????blocks??????
    for i in range(num_nodes):
        for j in range(max(0, i - 2), min(num_nodes, i + 3)):  # ????????
            if i != j:
                sim = calculate_similarity(blocks[i], blocks[j])
                adjacency_matrix[i, j] = sim
                adjacency_matrix[j, i] = sim  # ???

    # ???????k???????
    for i in range(num_nodes):
        neighbors = np.argsort(-adjacency_matrix[i])[:k + 1]  # ?????k????,+1??????
        adjacency_matrix[i] = 0  # ???
        adjacency_matrix[i, neighbors] = 1  # ?????k????1

    return adjacency_matrix


def build_adjacency_matrices(dataloader, block_size, k=5):
    """
    ?????dataloader,???batch?????????????????
    :param dataloader: ??? DataLoader,????????? batchsize*3*640*360
    :param block_size: ?????
    :param k: ???????
    :return: ????????
    """
    all_batches_adj_matrices = []

    for batch_images, _ in dataloader:
        # ????????????
        assert batch_images.shape[2] == 320 and batch_images.shape[3] == 320, "Image dimensions must be 640x360."

        batch_adj_matrices = []
        # ??????
        for image in batch_images:
            # ?????????build_adjacency_matrix??
            image_np = image.permute(1, 2, 0).numpy()  # ???360*640*3
            adj_matrix = build_adjacency_matrix(image_np, block_size, k)
            batch_adj_matrices.append(adj_matrix)

        # ????????????
        full_adj_matrix = np.block([[adj if i == j else np.zeros_like(adj)
                                     for j, adj in enumerate(batch_adj_matrices)]
                                    for i, adj in enumerate(batch_adj_matrices)])
        all_batches_adj_matrices.append(full_adj_matrix)

    return all_batches_adj_matrices
def dense_to_sparse(adj_matrix):
    src, dst = np.where(adj_matrix.cpu().numpy() != 0)
    edge_index_np = np.stack([src, dst])
    edge_index = torch.from_numpy(edge_index_np).long().to(adj_matrix.device)

    return edge_index

# Standard library imports
import numpy as np
import cv2

# Scientific computing imports
from skimage.util import view_as_blocks

# PyTorch imports
import torch

def calculate_similarity(block1, block2):
    # Convert blocks to grayscale
    block1_gray = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
    block2_gray = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)

    # Sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    block1_sobel = cv2.filter2D(block1_gray, -1, sobel_x) + cv2.filter2D(block1_gray, -1, sobel_y)
    block2_sobel = cv2.filter2D(block2_gray, -1, sobel_x) + cv2.filter2D(block2_gray, -1, sobel_y)

    # Gaussian operator
    gaussian_kernel = cv2.getGaussianKernel(ksize=3, sigma=1)
    block1_gaussian = cv2.filter2D(block1_gray, -1, gaussian_kernel)
    block2_gaussian = cv2.filter2D(block2_gray, -1, gaussian_kernel)

    # Combine Sobel and Gaussian similarities
    sobel_similarity = np.sum(np.abs(block1_sobel - block2_sobel))
    gaussian_similarity = np.sum(np.abs(block1_gaussian - block2_gaussian))

    # Final similarity score
    similarity = sobel_similarity + gaussian_similarity
    return similarity

def build_adjacency_matrix(image, block_size, k):
    # Calculate padding needed to make dimensions divisible by block_size
    height, width = image.shape[0], image.shape[1]
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size
    
    # Pad the image if needed
    if pad_height > 0 or pad_width > 0:
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')

    # ??????blocks
    blocks = view_as_blocks(image, block_shape=(block_size, block_size, 3))
    num_blocks_y, num_blocks_x = blocks.shape[0], blocks.shape[1]
    blocks = blocks.reshape(-1, block_size, block_size, 3)  # ??blocks??

    # ???????
    num_nodes = num_blocks_y * num_blocks_x
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # ????blocks??????
    for i in range(num_nodes):
        for j in range(max(0, i - 2), min(num_nodes, i + 3)):  # ????????
            if i != j:
                sim = calculate_similarity(blocks[i], blocks[j])
                adjacency_matrix[i, j] = sim
                adjacency_matrix[j, i] = sim  # ???

    # ???????k???????
    for i in range(num_nodes):
        neighbors = np.argsort(-adjacency_matrix[i])[:k + 1]  # ?????k????,+1??????
        adjacency_matrix[i] = 0  # ???
        adjacency_matrix[i, neighbors] = 1  # ?????k????1

    return adjacency_matrix


def build_adjacency_matrices(dataloader, block_size, k=5):
    """
    ?????dataloader,???batch?????????????????
    :param dataloader: ??? DataLoader,????????? batchsize*3*640*360
    :param block_size: ?????
    :param k: ???????
    :return: ????????
    """
    all_batches_adj_matrices = []

    for batch_images, _ in dataloader:
        # ????????????
        assert batch_images.shape[2] == 320 and batch_images.shape[3] == 320, "Image dimensions must be 640x360."

        batch_adj_matrices = []
        # ??????
        for image in batch_images:
            # ?????????build_adjacency_matrix??
            image_np = image.permute(1, 2, 0).numpy()  # ???360*640*3
            adj_matrix = build_adjacency_matrix(image_np, block_size, k)
            batch_adj_matrices.append(adj_matrix)

        # ????????????
        full_adj_matrix = np.block([[adj if i == j else np.zeros_like(adj)
                                     for j, adj in enumerate(batch_adj_matrices)]
                                    for i, adj in enumerate(batch_adj_matrices)])
        all_batches_adj_matrices.append(full_adj_matrix)

    return all_batches_adj_matrices
def dense_to_sparse(adj_matrix):
    src, dst = np.where(adj_matrix.cpu().numpy() != 0)
    edge_index_np = np.stack([src, dst])
    edge_index = torch.from_numpy(edge_index_np).long().to(adj_matrix.device)

    return edge_index


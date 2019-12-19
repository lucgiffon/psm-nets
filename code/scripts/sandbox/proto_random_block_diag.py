from palmnet.utils import create_random_block_diag


def main():
    # will create dim1 x dim2 matrix
    dim1 = 5
    dim2 = 6
    block_size = 2

    # assert dim1 % 2 == 0 and dim2 % 2 == 0, "dim1 and dim2 must be factors of sparsity factor"


    print(create_random_block_diag(dim1, dim2, block_size))
    print(create_random_block_diag(dim1, dim2, block_size, mask=True))
    print(create_random_block_diag(dim1, dim2, block_size, mask=True, greedy=False))

if __name__ == "__main__":
    main()
import numpy as np

def kernel_matrix_from_file(path, max_len=-1):
    """
    Builds the kernel matrix B from a file.
    File is assumed to store the vector b[i] in rows
    return B constructed as:
                                0,       i< j
                    B_{i,j} =
                                b_{i-j}, i>=j
    """
    kernelvec = np.loadtxt(path)[:max_len]
    kernel = np.zeros((len(kernelvec), len(kernelvec)))
    for i in range(len(kernelvec)):
        for j in range(len(kernelvec)):
            kernel[i, j] = kernelvec[i-j]
    del(kernelvec)
    return kernel

def kernel_from_list(kernel_path_list, max_len=-1):
    """
    Builds kernel matrix for a list of kernel corrections

    Each kernel correction is loaded by kernel_matrix_from_file, format information can be found in the help of that function
    """
    matrices = []
    for p in kernel_path_list:
        matrices.append(kernel_matrix_from_file(p, max_len))
    kernel = matrices[0]

    for m in matrices[1:]:
        kernel = np.dot(m, kernel)
    return kernel

for i in np.nditer(cov_mat.diagonal()):
    if i > 1:
        print(i)
        print(np.where(cov_mat.diagonal()== i))

K-Means Clustering Paralelizado

1. Descrição
    Este projeto implementa o algoritmo K-Means para pontos 2D, usando:
    - Versão sequencial (base: TheAlgorithms/C)
    - Paralelização CPU com OpenMP
    - Paralelização GPU com CUDA

2. Requisitos
    - GCC ≥ 9 com suporte a OpenMP
    - CUDA Toolkit ≥ 10.0 (se usar GPU)
    - make

3. Compilação
    Sequencial:
        gcc -O3 -o kmeans_seq k_means_clustering.c -lm
    OpenMP:
        gcc -O3 -fopenmp -o kmeans_omp k_means_clustering.c -lm
    CUDA (opcional):
        nvcc -O3 -o kmeans_cuda k_means_cuda.cu

4. Execução
    ./kmeans_seq wholesale_customers_data.csv
    ./kmeans_omp wholesale_customers_data.csv [num_threads]
    ./kmeans_cuda wholesale_customers_data.csv

5. Alterações para paralelização
    Comentários marcados com "OpenMP" e "CUDA” no código devem ser descomentados.

6. Link do código sequencial base
    https://github.com/TheAlgorithms/C/blob/master/machine_learning/k_means_clustering.c

7. Autores
    - Lakhan Nad (original)
    - Lucas Lopes (modificações OpenMP/CUDA)
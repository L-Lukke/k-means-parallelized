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

3. Pré-pricessamento

   - As colunas "Milk" e "Grocery" da base original foram extraídas com: 'cut -d',' -f4,5 "Wholesale customers data.csv" > "wholesale_customers_data.csv"'

5. Compilação

   - Sequencial:
        'gcc -O3 -o kmeans_seq k_means_clustering.c -lm'
   
    - OpenMP:
        'gcc -O3 -fopenmp -o kmeans_omp k_means_clustering.c -lm'
   
    - CUDA:
        'nvcc -O3 -o kmeans_cuda k_means_cuda.cu'

7. Execução
    - ./kmeans_seq wholesale_customers_data.csv
    - ./kmeans_omp wholesale_customers_data.csv [num_threads]
    - ./kmeans_cuda wholesale_customers_data.csv

8. Alterações para paralelização
    Comentários marcados com "OpenMP" e "CUDA” no código devem ser descomentados.

9. Link do código sequencial base
    https://github.com/TheAlgorithms/C/blob/master/machine_learning/k_means_clustering.c

10. Autores
    - Lakhan Nad (original)
    - Lucas Lopes (modificações OpenMP/CUDA)

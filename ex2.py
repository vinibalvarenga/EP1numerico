import numpy as np
import matplotlib.pyplot as plt

EPS = 10**(-15)
ITR = 50

def critSassenfeld():
    return

def critLinhas(matrizA):
    i = 0
    # Iteracao pelas linhas da matriz, somando os termos delas em modulo
    while i  < matrizA.shape[0]:
        somaLinha = 0
        j = 0
        while j < matrizA.shape[1]:   
            # Se o elemento nao for da diagonal principal, soma-se seu modulo
            if i != j:
                somaLinha = somaLinha + abs(matrizA[i][j])
            j = j + 1
        # Ao final da soma em uma linha, confere se o criterio das linhas esta satisfeito
        if somaLinha >= abs(matrizA[i][i]):
            print("linha: ", i)
            return False
        i = i + 1
    # se esta satisfeito, retorna true
    return True

# Retorna o autovetor associado ao maior autovalor de uma matriz
def calcXStar(A):
    eigenValues = np.linalg.eig(A)
    #seleciona o autovalor dominante
    lambda_n = np.min(eigenValues[0])
    i = 0
    for mi_k in eigenValues[0]:
        if mi_k == lambda_n:
            novaMatriz = eigenValues[1].T 
            return (novaMatriz[i,:]) 
        i = i + 1

# Retorna True se o citerio de parada foi satisfeito, not, caso contrario
def critParada(x_k, xStar):
    # min {|-xk + x*|, |xk+x*|}
    norm = min(np.linalg.norm(x_k - xStar), np.linalg.norm(x_k + xStar))
    if norm >= EPS:
        return True
    return False

def metodoSOR(A, b, omega, vetorInicial):
    k = 0
    x_k_sor = np.copy(vetorInicial)
    # residuo inicial
    r = np.linalg.norm(np.matmul(A, x_k_sor) - b)

    while r > EPS and k < ITR:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i, j] * x_k_sor[j]

            x_k_sor[i] = (1-omega) * x_k_sor[i] + (omega /A[i, i]) * (b[i] - sigma)
            
            r = np.linalg.norm( np.matmul(A, x_k_sor) - b)

            k = k + 1     
    return x_k_sor
   
#  Calcula o proximo autovalor a partir da formula, note que x_til = A_1_x_k
def proxAutoValor(x_k, A_1_x_k):

    if np.all( (x_k == 0) ):
        return 0

    numerador = np.inner(x_k.T, A_1_x_k)
    denominador = np.inner(x_k.T, x_k)

    if denominador != 0:
        return numerador/denominador

# Plota o gráfico com os erros 
def plotagem (erroAutovalor, erroAutovetor, erroAssintotico, erroAssintoticoQuadrado, k):
    plt.plot(erroAutovalor, 'k', erroAutovetor, 'g', erroAssintotico, 'b', erroAssintoticoQuadrado, 'r')
    plt.yscale('log')
    plt.xlabel('Interações')
    plt.ylabel('Erro (escala log)')
    plt.title('Dados do exercicio 2')
    plt.axis([1, k,  10**(-15), 1])

    plt.show()


# Retorna o menor autovalor de A, com seu autovetor associado
# caso nao for possivel resolver o sistema, retorna False
def metodoPotInversa(A, omega, inicial):
    #Seleciona os autovalores e autovetores dominantes para comparacao
    autovalores = np.sort(eigenValues[0])
    lambda_n = autovalores[0]
    lambda_n_1 = autovalores[1]

    xStar = calcXStar(A)

    #arrays de erro
    erroAutovetor = np.array([])
    erroAutovalor = np.array([])
    erroAssintotico = np.array([])
    erroAssintoticoQuadrado= np.array([])

    #arrays iniciais
    x_k = np.copy(inicial)
    x_k_1 = np.copy(inicial)
    
    # Antes de mais nada, checa o criterio das linhas
    if not critLinhas(A):
        print("Criterio das linhas nao satisfeito")
        return False
    # se for satisfeito, fazemos as iteracoes
    k=1
    while k <= ITR and critParada(x_k_1, xStar):

        x_k = np.copy(x_k_1)
        x_til = np.empty_like(x_k)

        # a cada iteracao achamos x_til com o metodo sor
        x_til = metodoSOR(A, np.copy(x_k), omega, np.random.random(A.shape[0]))

        if type(x_til) == bool:
            print("Metodo da potencia inverso abortado")
            print(k, x_til)
            return False

        if np.all( (x_til == 0) ):
            x_k_1 = np.zeros(A.shape[0])
        else:
            x_k_1 = (x_til) / (np.linalg.norm(x_til))

        # Lembrando que A-1x(k) =x(k+1), calculamos a proxima iteracao do autovalor
        mi_k = proxAutoValor(x_k, x_til) 

        #atualiza os arrays de erro
        erroAssintotico = np.append(erroAssintotico, (lambda_n/lambda_n_1)**(k))
        erroAssintoticoQuadrado = np.append(erroAssintoticoQuadrado, (lambda_n/lambda_n_1)**(2*k)) 
        erroAutovalor = np.append(erroAutovalor, [abs(1/mi_k - lambda_n)] )
        erroAutovetor = np.append(erroAutovetor, [ min(np.linalg.norm(x_k_1 - xStar), np.linalg.norm(x_k_1 + xStar)) ] )

        k = k + 1 
    plotagem (erroAutovalor, erroAutovetor, erroAssintotico, erroAssintoticoQuadrado, k)
    return np.array( [1/mi_k, x_k_1], dtype=object )



# Exercicio de teste 1

B = np.random.random((10, 10))
I = np.identity(10)
A = B + B.T + 10*I
eigenValues = np.linalg.eig(A)

eingen = metodoPotInversa(A, 1.15, np.random.random(10))


# Exercicio de teste 2

#2.1 
B0 = np.random.random((8, 8))
p = 200
C = p * np.identity(8)
B = B0 + C
B_1 = np.linalg.inv(B)
D = np.diag(np.arange(1,9))
A = np.dot(np.dot(B, D), B_1)
eigenValues = np.linalg.eig(A)
eingen = metodoPotInversa(A, 1, np.random.random(8))

#2.2
B0 = np.random.random((8, 8))
p = 200000
C = p * np.identity(8)
B = B0 + C
B_1 = np.linalg.inv(B)

D = np.array ([ [0.1, 0, 0, 0, 0, 0, 0, 0],
                [0, 200, 0, 0, 0, 0, 0, 0],
                [0, 0, 300, 0, 0, 0, 0, 0],
                [0, 0, 0, 400, 0, 0, 0, 0],
                [0, 0, 0, 0, 500, 0, 0, 0],
                [0, 0, 0, 0, 0, 600, 0, 0],
                [0, 0, 0, 0, 0, 0, 700, 0],
                [0, 0, 0, 0, 0, 0, 0, 800]
                ])

A = np.dot(np.dot(B, D), B_1)
eigenValues = np.linalg.eig(A)
eingen = metodoPotInversa(A, 1, np.random.random(8))
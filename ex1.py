import numpy as np
import matplotlib.pyplot as plt

#Epsilon
Ep = 10**(-14)

#calcula o autovetor da proxima iteracao
def calculaProxAutoVetor (vetorA, vetorX):
    vetorMult = vetorA @ vetorX 
    norma = np.linalg.norm(vetorMult)
    return vetorMult/norma

#calcula o autovalor da proxima iteracao
def calculaProxAutoValor(vetorA, vetorX):
    parteSuperior = vetorX.T @ vetorA @ vetorX
    parteInferior = vetorX.T @ vetorX
    return parteSuperior/parteInferior

#escreve o erro do autovalor e do modulo do autovetor sempre que chamada
def escreveErro(interacao):
    erroAutoValor.append(min(abs(autovalorLambda1-autovalor), abs(autovalorLambda1+autovalor)))
    erroAutoVetor.append(min(np.linalg.norm(autovetDominante-autovetor), np.linalg.norm(autovetDominante+autovetor)))
    erroAutoValores1e2.append(abs((autovalorLambda2/autovalorLambda1)**(interacao)))
    erroAutoValores1e2AoQuadrado.append(abs((autovalorLambda2/autovalorLambda1)**(2*interacao))) 
    
#gera um vetor de n linhas
def geraX_0(n):
    return np.random.random((n))

#seleciona o autovetor do autovalor dominante
def selecionaAutoVetor(A):
    eigenValues = np.linalg.eig(A)
    #seleciona o autovalor dominante
    lambda_n = np.max(eigenValues[0])
    i = 0
    for mi_k in eigenValues[0]:
        if mi_k == lambda_n:
            novaMatriz = eigenValues[1].T 
            return (novaMatriz[i,:]) 
        i = i + 1

#calcula o segundo autovalor dominante e o incide que ele aparece na variavel guardavaloresReais
def calculaLambda2(guardaValoresReais):
    autovalores = np.sort(guardaValoresReais[0])
    return autovalores[1]

#limpa listas de erros
def limpaErros():
    erroAutoValor.clear()
    erroAutoVetor.clear()
    erroAutoValores1e2.clear()
    erroAutoValores1e2AoQuadrado.clear() 


#variaveis para guardar os erros de cada iteracao
erroAutoValor = []
erroAutoVetor = []
erroAutoValores1e2 = []
erroAutoValores1e2AoQuadrado = [] 

#numero de iteracoes máximo
iteracoesMax = 70

#exercicio 1.1
B = []

#tamanho da matriz formada
n = 10

#gera matriz 10x10 e coloca numeros aleatorios de 0 a 1 na matriz
B = np.random.random((n, n))

#coloca o valor em A
A = B + B.T

#escreve a matriz utilizada para o exercicio 3.4
np.savetxt('ex1_1.txt', A, fmt='%f')

#gera um X inicial qualquer
x_0 = geraX_0(n)

#calcula os valores do autovetor e dos autovalores
guardaValoresReais = np.linalg.eig(A)

#seleciona o autovalor dominante (lambda1) e o segundo auto valor dominante (lambda2)
autovalorLambda1 = max(guardaValoresReais[0])
autovalorLambda2 = calculaLambda2(guardaValoresReais)

#seleciona os valores para o autovetor
autovetDominante = selecionaAutoVetor(A)

autovetor = x_0
i = 1

while (i<iteracoesMax and min(np.linalg.norm(autovetDominante+autovetor), np.linalg.norm(autovetDominante-autovetor))>Ep):
    
    #calcula o autovalor da interacao atual(mi_k)
    autovalor = calculaProxAutoValor (A, autovetor) 
    
    
    escreveErro(i)
    i = i + 1

    #calcula o proximo autovetor (xk+1)
    autovetor = calculaProxAutoVetor (A, autovetor)

#plotando o gráfico
plt.plot(erroAutoValor, 'k', erroAutoVetor, 'g', erroAutoValores1e2, 'b', erroAutoValores1e2AoQuadrado, 'r')
plt.yscale('log')
plt.xlabel('Interações')
plt.ylabel('Erro (escala log)')
plt.title('Erros do exercicio 1.1')
plt.axis([1, i,  10**(-15), 1])

plt.show()


#exercicio1.2
B=[]

#tamanho das matrizes
n = 5
iteracoesMax = 70

limpaErros()

#gera matriz 5x5 e coloca numeros aleatorios de 0 a 1 na matriz
B = np.random.random((n, n))

Binv = np.linalg.inv(B)

#lambda1 é 5 e lambda2 é 4 (pequena diferenca)
D1 = np.array([[5, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 1]])

#cria a matriz A
A1 = B @ D1 @ Binv

#escreve a matriz utilizada para o exercicio 3.4
np.savetxt('ex1_2.txt', A1, fmt='%f')
#gera um X inicial qualquer
x_0 = geraX_0(n)

#calcula os valores do autovetor e dos autovalores
guardaValoresReais = np.linalg.eig(A1)

#seleciona o autovalor dominante (lambda1) e o segundo auto valor dominante (lambda2)
autovalorLambda1 = max(guardaValoresReais[0])
autovalorLambda2 = calculaLambda2(guardaValoresReais)

#como sabe-se o autovalor neste caso, pode-se já coloca-lo
autovalorLambda1 = 5
autovalorLambda2 = 4

#seleciona os valores para o autovetor
autovetDominante = selecionaAutoVetor(A1)

autovetor = x_0
i = 1

while (i<iteracoesMax and min(np.linalg.norm(autovetDominante+autovetor), np.linalg.norm(autovetDominante-autovetor))>Ep):
    
    #calcula o autovalor da interacao atual(mi_k)
    
    autovalor = calculaProxAutoValor (A1, autovetor) 
    
    
    escreveErro(i)
    i = i + 1

    #calcula o proximo autovetor (xk+1)
    autovetor = calculaProxAutoVetor (A1, autovetor)
    


plt.figure()
plt.subplot(211)
#plotando o gráfico
plt.plot(erroAutoValor, 'k', erroAutoVetor, 'g', erroAutoValores1e2, 'b', erroAutoValores1e2AoQuadrado, 'r')
plt.yscale('log')
plt.title('Dados do exercicio 1.2')
plt.axis([1, i,  10**(-15), 1])

#segunda parte do exercicio 1.2
limpaErros()

#lambda1 é 20 e lambda2 é 4 (grande diferenca)
D2 = np.array([[20, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 1]])

#cria a matriz A2
A2 = B @ D2 @ Binv

#gera um X inicial qualquer
x_0 = geraX_0(n)


#calcula os valores do autovetor e dos autovalores
guardaValoresReais = np.linalg.eig(A2)

#seleciona o autovalor dominante (lambda1) e o segundo auto valor dominante (lambda2)
autovalorLambda1 = max(guardaValoresReais[0])
autovalorLambda2 = calculaLambda2(guardaValoresReais)

#como sabe-se o autovalor, pode-se já coloca-lo
autovalorLambda1 = 20
autovalorLambda2 = 4

#seleciona os valores para o autovetor
autovetDominante = selecionaAutoVetor(A2)

#primeira interacao dos calculos
autovetor = x_0
i = 1

while (i<iteracoesMax and min(np.linalg.norm(autovetDominante+autovetor), np.linalg.norm(autovetDominante-autovetor))>Ep):

    #calcula o autovalor da interacao atual(mi_k)
    
    autovalor = calculaProxAutoValor (A2, autovetor) 
    
    
    escreveErro(i)
    i = i + 1
    #calcula o proximo autovetor (xk+1)
    autovetor = calculaProxAutoVetor (A2, autovetor)

plt.subplot(212)
#plotando o gráfico
plt.plot(erroAutoValor, 'k', erroAutoVetor, 'g', erroAutoValores1e2, 'b', erroAutoValores1e2AoQuadrado, 'r')
plt.yscale('log')
plt.xlabel('Interações')
plt.ylabel('Erro (escala log)')
plt.axis([1, i,  10**(-15), 1])

plt.show()
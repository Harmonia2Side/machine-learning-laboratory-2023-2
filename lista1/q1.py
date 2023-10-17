# %% [markdown]
# ### Bibliotecas

# %%
# Instalação manual de bibliotecas
# !pip install numpy matplotlib pandas seaborn
# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# WARNING: N entendi o que baixar na 3C TODO ver isso depois
# CUDA. Lindo. Isso vai ser divertido: https://github.com/CannyLab/tsne-cuda

# %% [markdown]
# # 1) Dada a base de dados Haberman's Survival (disponibilizada em http://archive.ics.uci.edu/dataset/43/haberman+s+survival), obtenha:

# %% [markdown]
# Leitura dos dados

# %%

#Faz a leitura do arquivo com os dados
#header = None --> o arquivo não tem cabeçalho
#names --> coloca nomes para cada coluna
data = pd.read_csv('data/haberman.data', header = None, names = ['age', 'op_year', 'pos_nodes', 'survived'])
data.head()
# data.describe()

# %% [markdown]
# ## a) A média e variância de cada um dos atributos;

# %%
# seleciona apenas atributos de entrada
faixa = np.arange(0,3)
print(faixa)

# %%
# Imprime média e variância de cada atributo
print('Média')
print(data.iloc[:,faixa].mean())

print('\r\nVariância')
print(data.iloc[:,faixa].var())

# %% [markdown]
# ## b) A média e variância de cada um dos atributos para cada uma das classes;

# %%
# Como pegar atributos filtrados por classe
# Extrai todos os valores únicos e os coloca num vetor
classes = data['survived'].unique()
# 1 = the patient survived 5 years or longer
# 2 = the patient died within 5 year
print(classes)

for i in range(0,classes.size):
    data_select = data[data['survived'] == classes[i]]
    print(f"Média 'survived' == {classes[i]}\n")
    print(data_select.iloc[:,faixa].mean())
    print("\n")
    print(f"Variância 'survived' == {classes[i]}\n")
    print(data_select.iloc[:,faixa].var())
    print("\n\n")


# %% [markdown]
# ## c) A matriz de coeficientes de correlação;

# %%
#Usamos numpy para realizar os cálculos
eigValues, eigVectors = np.linalg.eig(data.iloc[:,faixa].corr())
print(f" {eigValues}")
print()
print(eigVectors)


# %% [markdown]
# ## d) O histograma com 8 bins de cada um dos atributos para cada uma das classes (gere gráficos dos histogramas com cores diferentes para cada classe)

# %%
# Imprime os histogramas
print('classes[0] => survived')
print('classes[1] => not survived')

# Títulos e labels de figura
figlabels = {
    "xlabel":"Idade", 
    "title":"Histograma de Idade de Pacientes"
}

# legenda
legend = ['survived', 'not survived']

# Cria o canvas
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()

# -- AGE --
# slice = pd.DataFrame()
# slice['survived'] = data[data['survived'] == classes[0]].iloc[:,0]
# slice['not-survived'] = data[data['survived'] == classes[1]].iloc[:,0]

# slice.plot.hist(bins=8, alpha=0.5, **figlabels)

# # survived
data[data['survived'] == classes[0]].iloc[:,0].plot.hist(bins=8, alpha=0.5)
# # not survived
data[data['survived'] == classes[1]].iloc[:,0].plot.hist(bins=8, alpha=0.5, **figlabels)

ax.legend(legend)

# %%

# -- OP_YEAR --
# Títulos e labels de figura

figlabels = {
    "xlabel":"Ano da Operação", 
    "title":"Histograma de Ano da Operação de Pacientes"
}

# Cria o canvas
# fig2 = plt.figure(figsize=(10,10))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()

# # survived
data[data['survived'] == classes[0]].iloc[:,1].plot.hist(bins=8, alpha=0.5)
# # not survived
data[data['survived'] == classes[1]].iloc[:,1].plot.hist(bins=8, alpha=0.5, **figlabels)

ax.legend(legend)

# %%
# -- pos_nodes --
# Títulos e labels de figura

figlabels = {
    "xlabel":"Number of positive axillary nodes detected", 
    "title":"Histograma de Nódulos de Pacientes"
}

# Cria o canvas
# fig3 = plt.figure(figsize=(10,10))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()

# # survived
data[data['survived'] == classes[0]].iloc[:,2].plot.hist(bins=8, alpha=0.5)
# # not survived
data[data['survived'] == classes[1]].iloc[:,2].plot.hist(bins=8, alpha=0.5, **figlabels)

ax.legend(legend)

# %% [markdown]
# ## e) Gere um gráfico 3D das amostras, identificando cada classe. Analise o gráfico e informe se a tarefa de classificação é fácil ou difícil e justifique a sua resposta.
# 

# %%
# Visualização 3D

# cria o canvas 3d
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')

colors = ['tab:blue' ,'tab:orange']

ax.set_xlabel('age')
ax.set_ylabel('op_year')
ax.set_zlabel('pos_nodes')

for i in range(0,classes.size):
    data_select = data[data['survived'] == classes[i]]
    ax.scatter(data_select['age'], data_select['op_year'], data_select['pos_nodes'], marker='o', c=colors[i])

ax.legend(legend)

plt.show()

# %% [markdown]
# A tarefa de classificação é difícil, pois não há correlação clara de agrupamento entre os vaores dos parâmetros e a classe de cada amostra. As amostras de sobreviventes e de não sobreviventes estão misturadas.



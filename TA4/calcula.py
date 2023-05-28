import pandas as pd

# Carregar o dataset em um DataFrame
df = pd.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Média, desvio padrão e moda de cada variável
media = df.mean(numeric_only=True)
desvio = df.std(numeric_only=True)
moda = df.mode(numeric_only=True).iloc[0]
print('Média:')
print(media)
print('\nDesvio padrão:')
print(desvio)
print('\nModa:')
print(moda)


# Frequência de cada categoria
frequencia = df['class'].value_counts()
print('\nFrequência de cada categoria:')
print(frequencia)

# Média, desvio padrão e moda de cada variável por categoria
media_por_categoria = df.groupby('class').mean()
desvio_por_categoria = df.groupby('class').std()
moda_por_categoria = df.groupby('class').apply(lambda x: x.mode().iloc[0])
print('\nMédia por categoria:')
print(media_por_categoria)
print('\nDesvio padrão por categoria:')
print(desvio_por_categoria)
print('\nModa por categoria:')
print(moda_por_categoria)
